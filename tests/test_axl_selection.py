import pyarrow as pa
import pytest
from data.axl.selection import group_of, load_tags


@pytest.mark.parametrize("tags,group", [
    ({"animal"}, "animal"), ({"mythical creature", "anthropomorphic"}, "animal"),
    ({"animal", "anthropomorphic"}, "animal"), ({"character"}, "humanoid"),
    ({"anthropomorphic", "toy"}, "humanoid"), ({"character", "weapon"}, "humanoid"),
    ({"weapon"}, None), ({"anatomy"}, None), ({"vehicle"}, None), (set(), None),
])
def test_group_of(tags, group):
    assert group_of(tags) == group


def test_load_tags_groups_rows_by_uuid():
    table = pa.table({"uuid": ["a", "a", "b"], "ordinal": [0, 1, 2], "category": ["character", "toy", "animal"]})
    assert load_tags(table) == {"a": {"character", "toy"}, "b": {"animal"}}
