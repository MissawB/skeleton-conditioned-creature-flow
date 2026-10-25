import os
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from data.axl.remote_parquet import open_parquet


def test_open_parquet_reads_a_local_file(tmp_path):
    path = tmp_path / "t.parquet"
    pq.write_table(pa.table({"uuid": ["a", "b", "c"]}), path)
    assert open_parquet(str(path)).metadata.num_rows == 3


@pytest.mark.skipif("AXL_NETWORK_TESTS" not in os.environ, reason="network test, set AXL_NETWORK_TESTS=1")
def test_open_parquet_reads_the_mirror_over_http():
    assert open_parquet("train/models.parquet").metadata.num_row_groups == 729


def test_range_reads_retry_after_network_errors(monkeypatch):
    import io
    import urllib.error
    from data.axl import remote_parquet

    calls = []

    class Response(io.BytesIO):
        headers = {"Content-Length": "10"}

        def geturl(self):
            return "https://cdn.example/file"

    def fake_urlopen(request, timeout=None):
        calls.append(timeout)
        if request.get_method() == "HEAD":
            return Response(b"")
        if len(calls) < 4:
            raise urllib.error.URLError("network unreachable")
        return Response(b"0123456789"[2:6])

    monkeypatch.setattr(remote_parquet.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(remote_parquet.time, "sleep", lambda s: None)
    f = remote_parquet.RangeFile("https://huggingface.co/x")
    f.seek(2)
    assert f.read(4) == b"2345"
    assert all(t is not None for t in calls)  # every request has a timeout, so a dead socket cannot hang the build
