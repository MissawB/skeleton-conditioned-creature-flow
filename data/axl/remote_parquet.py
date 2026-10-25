"""
Reads the Parquet mirror of Articulation-XL2.0 with HTTP Range requests, one row group at a time,
so the 25 GB mesh table never has to sit on disk.
"""
import io
import os
import time
import urllib.error
import urllib.request
import pyarrow.parquet as pq

MIRROR_URL = "https://huggingface.co/datasets/chibifire/datasource-articulation-xl2-parquet/resolve/main/"
TIMEOUT_S = 60
RETRIES = 6

def _fetch(request: urllib.request.Request):
    """(final url, headers, body) with a timeout, retried with backoff around both the request and the body read:
    a sleeping laptop or a dropped Wi-Fi must not hang or kill a long build."""
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                return response.geturl(), response.headers, response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == RETRIES - 1:
                raise
            time.sleep(min(2 ** attempt * 5, 120))

class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file; follows the Hub redirect once, then reuses the final URL."""

    def __init__(self, url: str):
        # A one-byte range, not HEAD: urllib turns a redirected HEAD into a GET of the whole file
        self.url, headers, _ = _fetch(urllib.request.Request(url, headers={"Range": "bytes=0-0"}))
        self.size = int(headers["Content-Range"].rsplit("/", 1)[1])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        _, _, data = _fetch(request)
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

def open_parquet(path: str) -> pq.ParquetFile:
    """Opens a local Parquet file, or a path relative to the mirror ("train/meshes.parquet")."""
    if os.path.exists(path):
        return pq.ParquetFile(path)
    return pq.ParquetFile(RangeFile(MIRROR_URL + path))
