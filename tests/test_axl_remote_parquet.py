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
