import torch
import torch.nn as nn
from typing import Tuple, List, Optional

class OptimalTransportFlowMatching:
    """Optimal Transport Flow Matching (OT-FM) framework with Euler and Heun ODE solvers."""
    
    def __init__(self, model: nn.Module):
        self.model = model

    def compute_loss(self, x_1: torch.Tensor, joints: torch.Tensor, x_0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes MSE loss along linear OT interpolants:
        x_t = (1 - t) * x_0 + t * x_1
        Target velocity u_t = x_1 - x_0
        x_0 may be supplied pre-coupled to x_1 (per-shape OT); otherwise it is drawn independently.
        """
        B, N, _ = x_1.shape
        device = x_1.device
        
        # Sample x_0 ~ N(0, I) and t ~ U[0, 1]
        if x_0 is None:
            x_0 = torch.randn_like(x_1)
        t = torch.rand(B, device=device)
        
        # OT interpolation
        t_expand = t.view(B, 1, 1)
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        target_v = x_1 - x_0
        
        pred_v = self.model(x_t, t, joints)
        loss = torch.mean((pred_v - target_v) ** 2)
        return loss

    @torch.no_grad()
    def sample(
        self,
        joints: torch.Tensor,
        num_points: int = 2048,
        num_steps: int = 20,
        solver: str = "heun",
        record_trajectory: bool = False
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Integrates dx/dt = v_theta(x, t, joints) from t=0 to t=1.
        """
        self.model.eval()
        B = joints.shape[0]
        device = joints.device
        
        x = torch.randn(B, num_points, 3, device=device)
        dt = 1.0 / num_steps
        
        trajectory = [x.clone().cpu()] if record_trajectory else []
        
        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.full((B,), t_val, device=device)
            
            if solver == "euler":
                v = self.model(x, t, joints)
                x = x + dt * v
            elif solver == "heun":
                v1 = self.model(x, t, joints)
                x_next_euler = x + dt * v1
                t_next = torch.full((B,), min(1.0, (i + 1) / num_steps), device=device)
                v2 = self.model(x_next_euler, t_next, joints)
                x = x + 0.5 * dt * (v1 + v2)
            else:
                raise ValueError(f"Unknown solver {solver}. Choose 'euler' or 'heun'.")
                
            if record_trajectory:
                trajectory.append(x.clone().cpu())
                
        return trajectory, x
