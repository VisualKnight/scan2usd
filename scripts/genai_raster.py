"""A pure-PyTorch stand-in for Hunyuan3D-2.1's `custom_rasterizer` CUDA extension (no compiler needed).

genai_generate.py registers this module as `custom_rasterizer` before importing Hunyuan3D's texture
pipeline. Same contract as the extension's rasterize/interpolate (hy3dpaint/custom_rasterizer):
clip-space vertices -> per-pixel 1-based face index (0 = empty) and perspective-correct
barycentrics, nearest depth wins, pixel centers sampled as in its CPU kernel. Vectorized on the GPU
over every (triangle, pixel-in-bounding-box) pair, in chunks.
"""

from __future__ import annotations

import torch

_FACE_BITS = 24  # depth token = quantized depth << 24 | face id (meshes here stay far below 16 M faces)


def _screen(V, W, H):
    w = V[:, 3]
    sx = (V[:, 0] / w * 0.5 + 0.5) * (W - 1) + 0.5
    sy = (0.5 + 0.5 * V[:, 1] / w) * (H - 1) + 0.5
    sz = V[:, 2] / w * 0.49999 + 0.5
    return sx, sy, sz, w


def _bary(ax, ay, bx, by, cx, cy, px, py):
    """calculateBarycentricCoordinate from the extension's rasterizer.h."""
    area = (cx - ax) * (by - ay) - (bx - ax) * (cy - ay)
    beta = ((cx - ax) * (py - ay) - (px - ax) * (cy - ay))
    gamma = ((px - ax) * (by - ay) - (bx - ax) * (py - ay))
    ok = area != 0
    inv = torch.where(ok, 1.0 / torch.where(ok, area, torch.ones_like(area)), torch.zeros_like(area))
    beta, gamma = beta * inv, gamma * inv
    alpha = 1.0 - beta - gamma
    bad = ~ok
    alpha, beta, gamma = (torch.where(bad, -torch.ones_like(alpha), alpha), torch.where(bad, -torch.ones_like(beta), beta),
                          torch.where(bad, -torch.ones_like(gamma), gamma))
    return alpha, beta, gamma


@torch.no_grad()
def rasterize(pos, tri, resolution, clamp_depth=None, use_depth_prior=0, chunk_pixels=24_000_000):
    V = pos[0].float()
    F = tri.long()
    H, W = int(resolution[0]), int(resolution[1])
    dev = V.device
    sx, sy, sz, w = _screen(V, W, H)
    fx, fy, fz = sx[F], sy[F], sz[F]  # (F, 3)
    x0 = torch.clamp(torch.floor(fx.min(1).values), 0, W - 1).long()
    x1 = torch.clamp(torch.floor(fx.max(1).values), 0, W - 1).long()
    y0 = torch.clamp(torch.floor(fy.min(1).values), 0, H - 1).long()
    y1 = torch.clamp(torch.floor(fy.max(1).values), 0, H - 1).long()
    valid = (fx.max(1).values >= 0) & (fx.min(1).values < W) & (fy.max(1).values >= 0) & (fy.min(1).values < H)
    bw, bh = (x1 - x0 + 1).clamp(min=0), (y1 - y0 + 1).clamp(min=0)
    count = torch.where(valid, bw * bh, torch.zeros_like(bw))
    big = torch.iinfo(torch.int64).max
    zbuf = torch.full((H * W,), big, dtype=torch.int64, device=dev)
    faces = torch.nonzero(count > 0).squeeze(1)
    csum = torch.cumsum(count[faces], 0)
    start = 0
    while start < len(faces):
        base = csum[start - 1] if start > 0 else torch.zeros((), dtype=csum.dtype, device=dev)
        end = int(torch.searchsorted(csum, base + chunk_pixels, right=True))
        end = max(end, start + 1)
        fsel = faces[start:end]
        n = count[fsel]
        fid = torch.repeat_interleave(fsel, n)
        offs = torch.cumsum(n, 0) - n
        local = torch.arange(int(n.sum()), device=dev) - torch.repeat_interleave(offs, n)
        px = x0[fid] + local % bw[fid]
        py = y0[fid] + torch.div(local, bw[fid], rounding_mode="floor")
        cx, cy = px.float() + 0.5, py.float() + 0.5
        a, b, c = _bary(fx[fid, 0], fy[fid, 0], fx[fid, 1], fy[fid, 1], fx[fid, 2], fy[fid, 2], cx, cy)
        inside = (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1) & (c >= 0) & (c <= 1)
        depth = a * fz[fid, 0] + b * fz[fid, 1] + c * fz[fid, 2]
        zq = (depth[inside] * (2 << 17)).long().clamp(min=0)
        token = (zq << _FACE_BITS) | (fid[inside] + 1)
        zbuf.scatter_reduce_(0, (py * W + px)[inside], token, reduce="amin")
        start = end
    hit = zbuf != big
    findices = torch.where(hit, zbuf & ((1 << _FACE_BITS) - 1), torch.zeros_like(zbuf)).view(H, W).int()
    bary = torch.zeros(H * W, 3, device=dev)
    pix = torch.nonzero(hit).squeeze(1)
    if len(pix):
        f = (zbuf[pix] & ((1 << _FACE_BITS) - 1)) - 1
        cx = (pix % W).float() + 0.5
        cy = torch.div(pix, W, rounding_mode="floor").float() + 0.5
        a, b, c = _bary(fx[f, 0], fy[f, 0], fx[f, 1], fy[f, 1], fx[f, 2], fy[f, 2], cx, cy)
        ww = w[F[f]]
        abc = torch.stack([a / ww[:, 0], b / ww[:, 1], c / ww[:, 2]], 1)  # perspective-correct, as the kernel
        bary[pix] = abc / abc.sum(1, keepdim=True)
    return findices, bary.view(H, W, 3)


def interpolate(col, findices, barycentric, tri):
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    result = torch.sum(result, axis=-2)
    return result.view(1, *result.shape)
