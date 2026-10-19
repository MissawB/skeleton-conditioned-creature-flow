import numpy as np
from typing import Set, List
from data.smal_quadrupeds import JOINT_NAMES, PARENT_INDICES

class KinematicHybridSplicer:
    """Combines subtrees from different animal skeletons to create chimeric skeletons."""
    
    def __init__(self):
        self.joint_names = JOINT_NAMES
        self.parents = PARENT_INDICES
        self.children = {i: [] for i in range(len(JOINT_NAMES))}
        for child, parent in enumerate(self.parents):
            if parent >= 0:
                self.children[parent].append(child)

    def _get_subtree_indices(self, root_joint_idx: int) -> Set[int]:
        """Collect all joint indices belonging to the subtree rooted at root_joint_idx."""
        subtree = set()
        queue = [root_joint_idx]
        while queue:
            curr = queue.pop(0)
            subtree.add(curr)
            queue.extend(self.children[curr])
        return subtree

    def splice(
        self,
        skel_host: np.ndarray,
        skel_donor: np.ndarray,
        junction_joint: str = "neck_base"
    ) -> np.ndarray:
        """
        Grafts the subtree of skel_donor rooted at junction_joint onto skel_host.
        """
        assert junction_joint in self.joint_names, f"Unknown junction joint: {junction_joint}"
        junction_idx = self.joint_names.index(junction_joint)
        
        subtree_indices = self._get_subtree_indices(junction_idx)
        # Exclude the junction root itself from translation offset base
        subtree_descendants = subtree_indices - {junction_idx}
        
        chimera = skel_host.copy()
        
        # Translation offset to align donor subtree with host junction position
        donor_junction_pos = skel_donor[junction_idx]
        host_junction_pos = skel_host[junction_idx]
        offset = host_junction_pos - donor_junction_pos
        
        for idx in subtree_descendants:
            chimera[idx] = skel_donor[idx] + offset
            
        return chimera.astype(np.float32)
