"""
Depth rendering utility using pyrender (replaces neural_renderer_pytorch).

This avoids the need for CUDA compilation at install time. Pyrender uses
OpenGL/EGL for offscreen rendering which is already a project dependency.
"""
import os
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import torch
import pyrender
import trimesh


def render_depth_map(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    focal_length: float,
    image_size: int,
) -> torch.Tensor:
    """
    Render a depth map for each mesh in a batch using pyrender.

    Replaces neural_renderer depth mode. The camera is at the origin
    looking along +Z (OpenGL convention handled internally by pyrender).

    Args:
        vertices: (B, V, 3) predicted vertices (already in camera space).
        faces:    (F, 3) face indices (shared across batch).
        focal_length: scalar focal length in pixels.
        image_size: output depth map is (image_size x image_size).

    Returns:
        depth: (B, H, W) torch.Tensor on the same device as vertices.
    """
    device = vertices.device
    B = vertices.shape[0]
    faces_np = faces.cpu().numpy()

    renderer = pyrender.OffscreenRenderer(
        viewport_width=image_size,
        viewport_height=image_size,
        point_size=1.0,
    )

    cx = cy = image_size / 2.0
    camera = pyrender.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=cx, cy=cy,
        znear=0.01, zfar=1e12,
    )

    # pyrender uses OpenGL convention (camera looks down -Z),
    # but the mesh vertices are in a coordinate system where the
    # subject is in front of the camera along +Z.  Flip Y and Z
    # to convert to OpenGL convention.
    flip = np.diag([1.0, -1.0, -1.0, 1.0])

    depth_maps = []
    for b in range(B):
        verts_np = vertices[b].detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)

        scene = pyrender.Scene()
        scene.add(mesh_pyrender)
        scene.add(camera, pose=flip)

        _, depth = renderer.render(scene)
        depth_maps.append(torch.tensor(depth, dtype=torch.float32, device=device))

    renderer.delete()
    return torch.stack(depth_maps, dim=0)  # (B, H, W)
