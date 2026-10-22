import torch
import torch.nn as nn
from typing import Tuple, List, Optional

class OptimalTransportFlowMatching:
    """Optimal Transport Flow Matching (OT-FM) framework with Euler and Heun ODE solvers."""
    
    def __init__(self, model: nn.Module, cond_drop_prob: float = 0.0):
        self.model = model
        # Share of training samples whose skeleton is dropped, enabling classifier-free guidance
        self.cond_drop_prob = cond_drop_prob

    def _velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        joints: torch.Tensor,
        guidance_scale: float,
        guide_model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        if guidance_scale == 1.0:
            return self.model(x, t, joints)
        if guide_model is not None:
            # Autoguidance: steer away from a weaker model's prediction, v = v_weak + w * (v - v_weak)
            v_weak = guide_model(x, t, joints)
            return v_weak + guidance_scale * (self.model(x, t, joints) - v_weak)
        # Conditional and unconditional passes in one batch: v = v_uncond + w * (v_cond - v_uncond)
        B = x.shape[0]
        cond_mask = torch.cat([torch.ones(B, dtype=torch.bool), torch.zeros(B, dtype=torch.bool)]).to(x.device)
        v = self.model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([joints, joints]), cond_mask=cond_mask)
        v_cond, v_uncond = v.chunk(2)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    def compute_loss(self, x_1: torch.Tensor, joints: torch.Tensor, x_0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes MSE loss along linear OT interpolants:
        x_t = (1 - t) * x_0 + t * x_1
        Target velocity u_t = x_1 - x_0
        x_0 may be supplied pre-coupled to x_1 (per-shape OT); otherwise it is drawn independently.
        """
        B, N, _ = x_1.shape
        device = x_1.device
        # Work in the network's coordinate frame (data std ~0.25 is scaled towards the noise std of 1)
        s = self.model.data_scale
        x_1, joints = x_1 * s, joints * s
        
        # Sample x_0 ~ N(0, I) and t ~ U[0, 1]
        if x_0 is None:
            x_0 = torch.randn_like(x_1)
        t = torch.rand(B, device=device)
        
        # OT interpolation
        t_expand = t.view(B, 1, 1)
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        target_v = x_1 - x_0
        
        cond_mask = None
        if self.cond_drop_prob > 0:
            cond_mask = torch.rand(B, device=device) >= self.cond_drop_prob
        pred_v = self.model(x_t, t, joints, cond_mask=cond_mask)
        loss = torch.mean((pred_v - target_v) ** 2)
        return loss

    @torch.no_grad()
    def sample(
        self,
        joints: torch.Tensor,
        num_points: int = 2048,
        num_steps: int = 20,
        solver: str = "heun",
        record_trajectory: bool = False,
        guidance_scale: float = 1.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        guide_model: Optional[nn.Module] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Integrates dx/dt = v_theta(x, t, joints) from t=0 to t=1.
        guidance_scale > 1 applies classifier-free guidance (needs a model trained with cond_drop_prob > 0),
        or autoguidance when guide_model (e.g. an early checkpoint of the same model) is given.
        Guidance is only applied for t in guidance_interval (Kynkaanniemi et al., 2024); elsewhere w = 1.
        """
        self.model.eval()
        if guide_model is not None:
            guide_model.eval()
            assert torch.equal(guide_model.data_scale, self.model.data_scale), "guide must share the data scale"
        s = self.model.data_scale
        joints = joints * s
        t_lo, t_hi = guidance_interval
        velocity = lambda x, t_val, t: self._velocity(
            x, t, joints, guidance_scale if t_lo <= t_val <= t_hi else 1.0, guide_model)
        B = joints.shape[0]
        device = joints.device
        
        x = torch.randn(B, num_points, 3, device=device)
        dt = 1.0 / num_steps
        
        trajectory = [(x / s).cpu()] if record_trajectory else []
        
        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.full((B,), t_val, device=device)
            
            if solver == "euler":
                v = velocity(x, t_val, t)
                x = x + dt * v
            elif solver == "heun":
                v1 = velocity(x, t_val, t)
                x_next_euler = x + dt * v1
                t_next_val = min(1.0, (i + 1) / num_steps)
                t_next = torch.full((B,), t_next_val, device=device)
                v2 = velocity(x_next_euler, t_next_val, t_next)
                x = x + 0.5 * dt * (v1 + v2)
            else:
                raise ValueError(f"Unknown solver {solver}. Choose 'euler' or 'heun'.")
                
            if record_trajectory:
                trajectory.append((x / s).cpu())
                
        return trajectory, x / s
