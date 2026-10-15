"""The pure-PyTorch stand-in for Hunyuan3D-2.1's CUDA rasterizer (scripts/genai_raster.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import genai_raster  # noqa: E402


def _clip(mesh):
    return torch.tensor(np.c_[mesh.vertices, np.ones(len(mesh.vertices))], dtype=torch.float32)[None]


def test_sphere_coverage_depth_and_barycentrics():
    m = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
    pos, tri = _clip(m), torch.tensor(m.faces, dtype=torch.int32)
    fi, bary = genai_raster.rasterize(pos, tri, (256, 256))
    hit = fi > 0
    # a disc of radius 0.5 in NDC [-1, 1]^2: pi * 0.25 / 4 of the image
    assert abs(hit.float().mean().item() - np.pi * 0.25 / 4) < 0.005
    assert torch.allclose(bary[hit].sum(-1), torch.ones(int(hit.sum())), atol=1e-5)
    z = genai_raster.interpolate(pos[..., 2:3], fi, bary, tri)[0, ..., 0]
    assert z[128, 128].item() == pytest.approx(-0.5, abs=0.01)  # the nearer (smaller depth) surface wins


def test_empty_and_offscreen():
    m = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    m.apply_translation([5.0, 0.0, 0.0])  # entirely off screen
    fi, bary = genai_raster.rasterize(_clip(m), torch.tensor(m.faces, dtype=torch.int32), (64, 48))
    assert fi.shape == (64, 48) and int((fi > 0).sum()) == 0 and bary.shape == (64, 48, 3)
