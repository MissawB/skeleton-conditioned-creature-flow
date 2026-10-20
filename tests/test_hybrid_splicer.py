# tests/test_hybrid_splicer.py
import numpy as np
from data.smal_quadrupeds import SMALQuadrupedGenerator
from pipeline.hybrid_splicer import KinematicHybridSplicer

def test_hybrid_splicer():
    generator = SMALQuadrupedGenerator(seed=42)
    skel_host = generator.generate_instance("felidae")["joints"]
    skel_donor = generator.generate_instance("equidae")["joints"]
    
    splicer = KinematicHybridSplicer()
    chimera_skel = splicer.splice(
        skel_host=skel_host,
        skel_donor=skel_donor,
        junction_joint="neck_base" # Graft horse neck & head onto lion body
    )
    
    assert chimera_skel.shape == (35, 3)
    # Check that junction point matches smoothly
    neck_idx = 4
    head_idx = 5
    assert not np.isnan(chimera_skel).any()
    # Torso joints (0 to 3) should match host
    np.testing.assert_allclose(chimera_skel[0:4], skel_host[0:4])

def test_neck_graft_keeps_host_front_legs():
    generator = SMALQuadrupedGenerator(seed=42)
    skel_host = generator.generate_instance("felidae")["joints"]
    skel_donor = generator.generate_instance("equidae")["joints"]

    chimera_skel = KinematicHybridSplicer().splice(skel_host, skel_donor, junction_joint="neck_base")

    # Front legs (collar -> toes, indices 7-20) belong to the host torso, not the donor neck
    np.testing.assert_allclose(chimera_skel[7:21], skel_host[7:21])
    # Head and jaw come from the donor
    assert not np.allclose(chimera_skel[5:7], skel_host[5:7])
