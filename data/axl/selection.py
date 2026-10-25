"""Which Articulation-XL models count as creatures, and on which side of the 70/30 animal/humanoid balance."""
from collections import defaultdict
from typing import Dict, Optional, Set
import pyarrow as pa

ANIMAL_TAGS = {"animal", "mythical creature"}
HUMANOID_TAGS = {"character", "anthropomorphic", "toy"}

def group_of(tags: Set[str]) -> Optional[str]:
    """'animal' if any animal tag (so 'animal, anthropomorphic' is an animal), 'humanoid' if any humanoid tag, else None.
    A creature that also carries an object tag ('character, weapon') is kept: it is a creature holding something."""
    if tags & ANIMAL_TAGS:
        return "animal"
    if tags & HUMANOID_TAGS:
        return "humanoid"
    return None

def load_tags(meta_categories: pa.Table) -> Dict[str, Set[str]]:
    tags = defaultdict(set)
    for uuid, category in zip(meta_categories.column("uuid").to_pylist(), meta_categories.column("category").to_pylist()):
        tags[uuid].add(category)
    return dict(tags)
