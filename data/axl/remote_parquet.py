"""
Reads the Parquet mirror of Articulation-XL2.0 with HTTP Range requests, one row group at a time,
so the 25 GB mesh table never has to sit on disk.
"""
import io
import os
import urllib.request
import pyarrow.parquet as pq

MIRROR_URL = "https://huggingface.co/datasets/chibifire/datasource-articulation-xl2-parquet/resolve/main/"

class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file; follows the Hub redirect once, then reuses the final URL."""

    def __init__(self, url: str):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            self.url = response.geturl()
            self.size = int(response.headers["Content-Length"])
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
        with urllib.request.urlopen(request) as response:
            data = response.read()
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
