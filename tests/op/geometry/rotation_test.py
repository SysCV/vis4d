"""Rotation tests."""
from __future__ import annotations

import itertools
import math
import unittest

import numpy as np
import torch

from vis4d.common.typing import NDArrayF64
from vis4d.op.geometry.rotation import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_multiply,
    quaternion_to_matrix,
)


# Testcases for rotation conversion adapted from:
# https://github.com/facebookresearch/pytorch3d/blob/main/tests/test_rotation_conversions.py
class TestRotationFuncs(unittest.TestCase):
    """Testcases for rotation utility functions."""

    @staticmethod
    def _tait_bryan_conventions() -> map[str]:
        """Get tait bryan conventions."""
        return map(  # pylint: disable=bad-builtin
            "".join, itertools.permutations("XYZ")
        )

    @staticmethod
    def _proper_euler_conventions() -> tuple[str]:
        """Get proper euler conventions."""
        letterpairs = itertools.permutations("XYZ", 2)
        return (l0 + l1 + l0 for l0, l1 in letterpairs)  # type: ignore

    def _all_euler_angle_conventions(self) -> tuple[str]:
        """Get all euler angles conventions."""
        return itertools.chain(  # type: ignore
            self._tait_bryan_conventions(), self._proper_euler_conventions()
        )

    def _assert_quaternions_close(
        self,
        quat: torch.Tensor | NDArrayF64,
        other: torch.Tensor | NDArrayF64,
        *,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
        msg: str | None = None,
    ) -> None:
        """Assert that two quaternions are (almost) equal."""
        self.assertEqual(np.shape(quat), np.shape(other))
        dot = (quat * other).sum(-1)
        ones = torch.ones_like(dot)
        self.assertTrue(
            torch.isclose(
                dot.abs(), ones, rtol=rtol, atol=atol, equal_nan=equal_nan
            ).all(),
            msg=msg,
        )

    def test_from_euler(self) -> None:
        """Euler -> mtx -> euler."""
        n_repetitions = 10
        # tolerance is how much we keep the middle angle away from the extreme
        # allowed values which make the calculation unstable (Gimbal lock).
        tolerance = 0.04
        half_pi = math.pi / 2
        data = torch.zeros(n_repetitions, 3)
        data.uniform_(-math.pi, math.pi)

        data[:, 1].uniform_(-half_pi + tolerance, half_pi - tolerance)
        for convention in self._tait_bryan_conventions():
            matrices = euler_angles_to_matrix(data, convention)
            mdata = matrix_to_euler_angles(matrices, convention)
            self.assertTrue(torch.isclose(data, mdata).all())

        data[:, 1] += half_pi
        for convention in self._proper_euler_conventions():
            matrices = euler_angles_to_matrix(data, convention)
            mdata = matrix_to_euler_angles(matrices, convention)
            self.assertTrue(torch.isclose(data, mdata, rtol=1e-4).all())

    def test_to_euler(self) -> None:
        """Mtx -> euler -> mtx."""
        data = random_rotations(13, dtype=torch.float64)
        for convention in self._all_euler_angle_conventions():
            euler_angles = matrix_to_euler_angles(data, convention)
            mdata = euler_angles_to_matrix(euler_angles, convention)
            self.assertTrue(torch.isclose(data, mdata).all())

    def test_from_quat(self) -> None:
        """Quat -> mtx -> quat."""
        data = random_quaternions(13, dtype=torch.float64)
        mdata = matrix_to_quaternion(quaternion_to_matrix(data))
        self._assert_quaternions_close(data, mdata)

    def test_to_quat(self) -> None:
        """Mtx -> quat -> mtx."""
        data = random_rotations(13, dtype=torch.float64)
        mdata = quaternion_to_matrix(matrix_to_quaternion(data))
        self.assertTrue(torch.isclose(data, mdata).all())

    def test_quat_grad_exists(self) -> None:
        """Quaternion calculations are differentiable."""
        rotation = random_rotations(1)[0]
        rotation.requires_grad = True
        modified = quaternion_to_matrix(matrix_to_quaternion(rotation))
        [g] = torch.autograd.grad(modified.sum(), rotation)
        self.assertTrue(torch.isfinite(g).all())

    def test_quaternion_multiplication(self) -> None:
        """Quaternion and matrix multiplication are equivalent."""
        a = random_quaternions(15, torch.float64).reshape((3, 5, 4))
        b = random_quaternions(21, torch.float64).reshape((7, 3, 1, 4))
        ab = quaternion_multiply(a, b)
        self.assertEqual(ab.shape, (7, 3, 5, 4))
        a_matrix = quaternion_to_matrix(a)
        b_matrix = quaternion_to_matrix(b)
        ab_matrix = torch.matmul(a_matrix, b_matrix)
        ab_from_matrix = matrix_to_quaternion(ab_matrix)
        self._assert_quaternions_close(ab, ab_from_matrix)

    def test_matrix_to_quaternion_corner_case(self) -> None:
        """Check no bad gradients from sqrt(0)."""
        matrix = torch.eye(3, requires_grad=True)
        target = torch.Tensor([0.984808, 0, 0.174, 0])

        optimizer = torch.optim.Adam([matrix], lr=0.05)
        optimizer.zero_grad()
        q = matrix_to_quaternion(matrix)
        loss = torch.sum((q - target) ** 2)
        loss.backward()
        optimizer.step()

        self.assertTrue(
            torch.isclose(matrix, matrix).all(),
            msg="Result has non-finite values",
        )
        delta = 1e-2
        self.assertLess(
            matrix.trace(),
            3.0 - delta,
            msg="Identity initialisation unchanged by a gradient step",
        )

    def test_quaternion_application(self) -> None:
        """Applying a quaternion is the same as applying the matrix."""
        quaternions = random_quaternions(3, torch.float64)
        quaternions.requires_grad = True
        matrices = quaternion_to_matrix(quaternions)
        points = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        transform1 = quaternion_apply(quaternions, points)
        transform2 = torch.matmul(matrices, points[..., None])[..., 0]
        self.assertTrue(torch.isclose(transform1, transform2).all())

        [p, q] = torch.autograd.grad(transform1.sum(), [points, quaternions])
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue(torch.isfinite(q).all())


def _copysign(tensor: torch.Tensor, sign_tensor: torch.Tensor) -> torch.Tensor:
    """Copy sign function.

    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        tensor: source tensor.
        sign_tensor: tensor whose signs will be used, same shape as 'tensor'.

    Returns:
        Tensor of the same shape as tensor with the signs of sign_tensor.
    """
    signs_differ = (tensor < 0) != (sign_tensor < 0)
    return torch.where(signs_differ, -tensor, tensor)


def random_quaternions(
    num: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate random quaternions representing rotations.

    Args:
        num: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default: uses the current
        device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    o = torch.randn((num, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    num: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate random rotations as 3x3 rotation matrices.

    Args:
        num: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default: uses the current
        device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(num, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)
