import math
import time
from typing import Callable

import numpy as np
import torch
import torch.optim as optim
import trimesh
from src.utils import libmcubes
from src.utils.libmise import MISE
from src.utils.libsimplify import simplify_mesh
from torch import autograd
from tqdm import tqdm, trange

from vis4d.common.typing import NDArrayBool, NDArrayF64
from vis4d.model.occupancy.conv_occnet import *

counter = 0

"""
  vis_n_outputs: 2
  refine: false
  n_x: 128
  n_z: 1
  batch_size: 100000
  refinement_step: 0
  vis_n_outputs: 30
  generate_mesh: true
  generate_pointcloud: true
  generation_dir: generation
  use_sampling: false
  resolution_0: 32
  upsampling_steps: 2
  simplify_nfaces: null
  copy_groundtruth: false
  copy_input: true
  latent_number: 4
  latent_H: 8
  latent_W: 8
  latent_ny: 2
  latent_nx: 2
  latent_repeat: true
  sliding_window: False # added for crop generation
"""


def make_3d_grid(bb_min, bb_max, shape):
    """Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    """
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


class Generator3D(object):
    """Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
        vol_info (dict): volume infomation
        vol_bound (dict): volume boundary
        simplify_nfaces (int): number of faces the mesh should be simplified to
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], NDArrayF64],
        points_batch_size=100000,
        resolution0=16,
        padding=0.1,
    ):
        self.model = model
        self.points_batch_size = points_batch_size
        self.padding = padding
        self.resolution0 = resolution0

    def generate_mesh(self, volume_bounds=np.ndarray):
        """Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        """
        box_size = 1 + self.padding
        # Shortcut
        nx = self.resolution0
        pointsf = box_size * make_3d_grid((-0.5,) * 3, (0.5,) * 3, (nx,) * 3)
        sdf_values = self.eval_points(pointsf)
        sdf_values = sdf_values.reshape(nx, nx, nx)  # type: ignore

        # Some short hands
        n_x, n_y, n_z = sdf_values.shape
        box_size = 1 + self.padding
        threshold = 0  # np.log(self.threshold) - np.log(1.0 - self.threshold)

        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            sdf_values, 1, "constant", constant_values=-1e6
        )
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold
        )

        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1

        # Normalize to bounding box
        vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
        vertices = box_size * (vertices - 0.5)
        normals = None

        # Create mesh
        mesh = trimesh.Trimesh(
            vertices, triangles, vertex_normals=normals, process=False
        )

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh
        return mesh

    def eval_points(self, p: torch.Tensor) -> NDArrayF64:
        """Evaluates the occupancy values for the points.

        Args:
            p (tensor): points
            c (tensor): encoded feature volumes
        """
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
            occupancy = self.model(pi.unsqueeze(0))
            occ_hats.append(occupancy)

        occ_hat = np.concatenate(occ_hats, axis=0)
        return occ_hat
