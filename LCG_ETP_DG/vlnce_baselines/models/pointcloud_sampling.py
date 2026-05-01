from __future__ import annotations

import torch


def denormalize_depth_to_meters(
    depth_norm: torch.Tensor,
    normalize_depth: bool,
    min_depth: float,
    max_depth: float,
) -> torch.Tensor:
    depth_norm = depth_norm.to(dtype=torch.float32)
    if not normalize_depth:
        return depth_norm
    return depth_norm * (max_depth - min_depth) + min_depth


def depth_to_pointcloud(
    depth_m: torch.Tensor,
    hfov_deg: float,
) -> torch.Tensor:
    """
    Convert a single perspective depth map to point cloud in camera coordinates.

    Args:
        depth_m: (H, W) depth in meters
        hfov_deg: horizontal field of view in degrees

    Returns:
        points: (N, 3) where z is forward depth.
    """
    assert depth_m.ndim == 2, f"depth_m must be (H,W), got {depth_m.shape}"
    h, w = depth_m.shape
    hfov = torch.deg2rad(torch.tensor(hfov_deg, dtype=torch.float32, device=depth_m.device))
    fx = (w * 0.5) / torch.tan(hfov * 0.5)
    fy = fx
    cx = (w - 1.0) * 0.5
    cy = (h - 1.0) * 0.5

    u = torch.arange(w, dtype=torch.float32, device=depth_m.device).unsqueeze(0).expand(h, w)
    v = torch.arange(h, dtype=torch.float32, device=depth_m.device).unsqueeze(1).expand(h, w)
    z = depth_m
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    valid = torch.isfinite(pts).all(dim=1) & (pts[:, 2] > 0)
    return pts[valid]


def crop_forward_depth(points: torch.Tensor, max_depth_m: float) -> torch.Tensor:
    if points.numel() == 0:
        return points
    mask = (points[:, 2] > 0.0) & (points[:, 2] <= float(max_depth_m))
    return points[mask]


def fps(points: torch.Tensor, n_samples: int, seed: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) with deterministic RNG.
    Complexity: O(N*K). Use only for relatively small point sets.
    """
    n = int(points.shape[0])
    if n == 0:
        return torch.zeros((n_samples, 3), dtype=torch.float32, device=points.device)

    generator = torch.Generator(device=points.device)
    generator.manual_seed(int(seed))

    if n <= n_samples:
        idx = torch.randint(0, n, size=(n_samples,), generator=generator, device=points.device)
        return points[idx].to(dtype=torch.float32)

    selected = torch.zeros(n_samples, dtype=torch.long, device=points.device)
    selected[0] = torch.randint(0, n, size=(1,), generator=generator, device=points.device)[0]

    dist = torch.full((n,), float("inf"), dtype=torch.float32, device=points.device)
    for i in range(1, n_samples):
        last = points[selected[i - 1]]
        d = torch.sum((points - last) ** 2, dim=1)
        dist = torch.minimum(dist, d)
        selected[i] = torch.argmax(dist)
    return points[selected].to(dtype=torch.float32)


def sample_pointcloud_from_depth(
    depth_norm_map: torch.Tensor,
    *,
    hfov_deg: float,
    normalize_depth: bool,
    min_depth: float,
    max_depth: float,
    enable_spatial_crop: bool,
    max_depth_m: float,
    num_points: int,
    seed: int,
) -> torch.Tensor:
    """
    Produce a sampled point cloud for one depth map.

    Returns:
        sampled_points: (num_points, 3) on the same device as input tensor.
    """
    if depth_norm_map.ndim == 3 and depth_norm_map.shape[-1] == 1:
        depth_norm_map = depth_norm_map[..., 0]
    assert depth_norm_map.ndim == 2, f"depth_norm_map must be (H,W) or (H,W,1), got {depth_norm_map.shape}"

    depth_norm = depth_norm_map.detach().to(dtype=torch.float32)
    depth_m = denormalize_depth_to_meters(
        depth_norm=depth_norm,
        normalize_depth=normalize_depth,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    pts = depth_to_pointcloud(depth_m=depth_m, hfov_deg=hfov_deg)
    if enable_spatial_crop:
        pts = crop_forward_depth(points=pts, max_depth_m=max_depth_m)
    sampled = fps(points=pts, n_samples=num_points, seed=seed)
    return sampled


def build_depth_feats_maps(observations: dict, num_imgs: int = 12) -> torch.Tensor:
    """
    Build a (B, num_imgs, H, W, 1) depth tensor aligned with PolicyViewSelectionETP ETP.forward(mode='waypoint').
    """
    depth_ref = None
    if "depth" in observations:
        depth_ref = observations["depth"]
    else:
        for k, v in observations.items():
            if "depth" in k and isinstance(v, torch.Tensor):
                depth_ref = v
                break
    if depth_ref is None:
        raise KeyError("No depth tensor found in observations.")

    b, h, w, c = depth_ref.shape
    depth_batch = torch.zeros_like(depth_ref).repeat(num_imgs, 1, 1, 1)

    a_count = 0
    for k, v in observations.items():
        if not isinstance(v, torch.Tensor):
            continue
        if "depth" not in k:
            continue
        for bi in range(v.size(0)):
            ra_count = (num_imgs - a_count) % num_imgs
            depth_batch[ra_count + bi * num_imgs] = v[bi]
        a_count += 1

    depth_maps = depth_batch.view(b, num_imgs, h, w, c)

    depth_feats_maps = torch.cat(
        (depth_maps[:, 0:1, ...], torch.flip(depth_maps[:, 1:, ...], [1])),
        dim=1,
    )
    return depth_feats_maps
