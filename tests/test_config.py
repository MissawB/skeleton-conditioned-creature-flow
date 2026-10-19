# tests/test_config.py
import os
import yaml

def test_default_config_exists_and_valid():
    config_path = os.path.join("configs", "default_config.yaml")
    assert os.path.exists(config_path), f"Config file not found at {config_path}"
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    assert "data" in cfg
    assert "model" in cfg
    assert "training" in cfg
    assert "sampling" in cfg
    
    assert cfg["data"]["num_points"] == 2048
    assert cfg["data"]["num_joints"] == 35
    assert cfg["model"]["hidden_dim"] == 256
    assert cfg["model"]["num_layers"] == 6
