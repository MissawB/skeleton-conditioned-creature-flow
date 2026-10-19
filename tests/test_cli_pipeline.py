# tests/test_cli_pipeline.py
import json
import os
import torch
import numpy as np
from data.smal_quadrupeds import SMALQuadrupedGenerator
from models.point_dit import PointDiT
from models.flow_matching import OptimalTransportFlowMatching
from pipeline.hybrid_splicer import KinematicHybridSplicer
from pipeline.metrics import compute_chamfer_distance

def test_sampling_and_export_json(tmp_path):
    model = PointDiT(hidden_dim=64, num_layers=2, num_heads=2)
    fm = OptimalTransportFlowMatching(model)
    generator = SMALQuadrupedGenerator(seed=42)
    
    skel = generator.generate_instance("felidae")["joints"]
    skel_t = torch.from_numpy(skel).unsqueeze(0)
    
    traj, x_final = fm.sample(skel_t, num_points=128, num_steps=5, record_trajectory=True)
    
    export_data = {
        "species": "lion",
        "joints": skel.tolist(),
        "trajectory": [step.squeeze(0).numpy().tolist() for step in traj]
    }
    
    out_file = os.path.join(tmp_path, "creatures.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"creatures": [export_data]}, f)
        
    assert os.path.exists(out_file)
    with open(out_file, "r") as f:
        loaded = json.load(f)
    assert len(loaded["creatures"]) == 1
    assert len(loaded["creatures"][0]["trajectory"]) == 6
