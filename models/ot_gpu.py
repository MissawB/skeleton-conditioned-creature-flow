"""
Per-shape optimal transport between noise points and surface points, computed on the GPU during training.
Replaces exact Hungarian matching, which costs seconds per shape on the CPU and cannot be precomputed for
tens of thousands of deformed creatures. Entropic Sinkhorn gives a near-optimal plan; it is then rounded to
an exact permutation so the matched noise stays a reordering of N(0, I) samples.
"""
import math
import torch

ROUNDING_ROUNDS = 20
CHUNK = 32  # blocks per Sinkhorn call: (32, 512, 512) plans take 32 MB in fp32

@torch.no_grad()
def _log_plan(x0: torch.Tensor, x1: torch.Tensor, eps: float, iters: int, eps_start: float = 0.3) -> torch.Tensor:
    """
    (B, N, N) log transport plan, rows = noise points, columns = data points, uniform marginals.
    eps decreases geometrically from eps_start to eps (potentials kept in cost units): a fixed small eps needs
    ~300 iterations to converge, the annealed schedule ~50.
    """
    cost = torch.cdist(x0.float(), x1.float()) ** 2
    cost = cost / cost.mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    log_marginal = -math.log(x0.shape[1])
    f = torch.zeros(x0.shape[:2], device=x0.device)
    g = torch.zeros_like(f)
    for e in torch.logspace(math.log10(eps_start), math.log10(eps), iters).tolist():
        f = e * log_marginal - e * torch.logsumexp((g[:, None, :] - cost) / e, dim=2)
        g = e * log_marginal - e * torch.logsumexp((f[:, :, None] - cost) / e, dim=1)
    return (f[:, :, None] + g[:, None, :] - cost) / eps

@torch.no_grad()
def _round_to_permutation(log_plan: torch.Tensor) -> torch.Tensor:
    """perm[b, j] = noise index matched to data point j; each round, free data points bid for their best free noise."""
    B, N, _ = log_plan.shape
    device = log_plan.device
    perm = torch.full((B, N), -1, dtype=torch.long, device=device)
    noise_free = torch.ones(B, N, dtype=torch.bool, device=device)
    data_free = torch.ones(B, N, dtype=torch.bool, device=device)
    columns = torch.arange(N, device=device).expand(B, N)
    neg_inf = torch.tensor(float("-inf"), device=device)
    for _ in range(ROUNDING_ROUNDS):
        if not data_free.any():
            break
        scores = log_plan.masked_fill(~noise_free[:, :, None], float("-inf"))
        best_val, best_noise = scores.max(dim=1)                           # (B, N) per data point
        bid = torch.where(data_free, best_val, neg_inf)
        top_bid = torch.full((B, N), float("-inf"), device=device).scatter_reduce(1, best_noise, bid, "amax")
        candidate = data_free & (bid > neg_inf) & (bid == top_bid.gather(1, best_noise))
        # Equal bids for the same noise point: the highest data index wins
        winner = torch.full((B, N), -1, dtype=torch.long, device=device).scatter_reduce(
            1, best_noise, torch.where(candidate, columns, -1), "amax")
        won = candidate & (winner.gather(1, best_noise) == columns)
        perm[won] = best_noise[won]
        noise_free[torch.arange(B, device=device)[:, None].expand(B, N)[won], best_noise[won]] = False
        data_free &= ~won
    for b in torch.where(data_free.any(1))[0].tolist():
        perm[b, data_free[b]] = torch.where(noise_free[b])[0]
    return perm

@torch.no_grad()
def sinkhorn_permutation(x0: torch.Tensor, x1: torch.Tensor, eps: float = 0.01, iters: int = 20, num_blocks: int = 4) -> torch.Tensor:
    """
    (B, N) permutation such that x0[b, perm[b, j]] is the transport partner of x1[b, j].
    With num_blocks > 1, noise and data points are split at random into equal blocks matched independently:
    each block is an exact OT problem between subsamples, num_blocks times cheaper (1.09x the exact cost
    instead of 1.04x at N = 2048 and 4 blocks, for a quarter of the time).
    """
    B, N, _ = x0.shape
    k = max(1, min(num_blocks, N))
    while N % k:
        k -= 1
    noise_order = torch.stack([torch.randperm(N, device=x0.device) for _ in range(B)])
    data_order = torch.stack([torch.randperm(N, device=x0.device) for _ in range(B)]) if k > 1 else noise_order.new_tensor(range(N)).expand(B, N)
    if k == 1:
        noise_order = data_order
    gather = lambda x, order: torch.gather(x, 1, order[..., None].expand_as(x)).reshape(B * k, N // k, 3)
    a, b = gather(x0, noise_order), gather(x1, data_order)
    local = torch.cat([_round_to_permutation(_log_plan(a[i:i + CHUNK], b[i:i + CHUNK], eps, iters))
                       for i in range(0, len(a), CHUNK)])                        # (B*k, N/k), index within block
    offsets = (torch.arange(k, device=x0.device) * (N // k)).repeat(B)[:, None]
    noise_slot = (local + offsets).reshape(B, N)                                   # position in noise_order
    perm = torch.empty_like(noise_slot)
    perm.scatter_(1, data_order, torch.gather(noise_order, 1, noise_slot))
    return perm

def ot_match(x0: torch.Tensor, x1: torch.Tensor, **kwargs) -> torch.Tensor:
    """x0 reordered so that point j of the result is paired with point j of x1."""
    perm = sinkhorn_permutation(x0, x1, **kwargs)
    return torch.gather(x0, 1, perm[..., None].expand_as(x0))
