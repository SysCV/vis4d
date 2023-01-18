"""Vis4D Visualization tools for analysis and debugging.

This file contains the legacy visualization tools that are not used anymore.
"""
from __future__ import annotations

from multiprocessing import Process
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches as mpatches
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from PIL import Image, ImageDraw
from torch import Tensor

from vis4d.common import NDArrayF64, NDArrayUI8
from vis4d.common.imports import DASH_AVAILABLE, PLOTLY_AVAILABLE
from vis4d.data.const import AxisMode
from vis4d.op.box.box3d import boxes3d_to_corners
from vis4d.op.geometry.projection import points_inside_image, project_points
from vis4d.vis.image.util import preprocess_image
from vis4d.vis.util import DEFAULT_COLOR_MAPPING

if DASH_AVAILABLE and PLOTLY_AVAILABLE:
    import dash
    import plotly.graph_objects as go
    from dash import dcc, html

ImageType = Union[torch.Tensor, NDArrayUI8, NDArrayF64]

ColorType = Union[
    Union[tuple[int], str],
    list[Union[tuple[int], str]],
    list[list[Union[tuple[int], str]]],
]


def draw_bev_canvas(
    x_min: int = -55,
    x_max: int = 55,
    y_min: int = -55,
    y_max: int = 55,
    fig_size: int = 10,
    dpi: int = 100,
    interval: int = 10,
) -> tuple[Figure, Axis]:
    """Draws a bird's eye view canvas.

    Draws a bird's eye view canvas with a car in the center and circular rings
    around it. Also plots the hardcoded camera poses.

    Args:
        x_min (int, optional): Minimum x value. Defaults to -55.
        x_max (int, optional): Maximum x value. Defaults to 55.
        y_min (int, optional): Minimum y value. Defaults to -55.
        y_max (int, optional): Maximum y value. Defaults to 55.
        fig_size (int, optional): Figure size. Defaults to 10.
        dpi (int, optional): DPI. Defaults to 100.
        interval (int, optional): Interval between rings. Defaults to 10.

    Returns:
        fig, ax: Figure and axis.
    """
    # Create canvas
    # sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.axis("off")
    # for key, spine in ax.spines.items():
    #    spine.set_visible(False)

    # Set x, y limit and mark border
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)
    ax.tick_params(axis="both", labelbottom=False, labelleft=False)
    ax.xaxis.set_minor_locator(MultipleLocator(interval))
    ax.yaxis.set_minor_locator(MultipleLocator(interval))

    for radius in range(y_max, -1, -interval):
        # Mark all around sector
        ax.add_patch(  # Draws circle around car.
            mpatches.Wedge(
                center=[0, 0],
                alpha=0.1,
                aa=True,
                r=radius,
                theta1=-180,
                theta2=180,
                fc="black",
            )
        )

        # Mark range
        if radius / np.sqrt(2) + 8 < x_max:
            ax.text(
                radius / np.sqrt(2) + 3,
                radius / np.sqrt(2) - 5,
                f"{radius}m",
                rotation=-45,
                color="darkblue",
                fontsize="xx-large",
            )

    # Mark camera FOVs
    # front
    ax.add_patch(
        mpatches.Wedge(
            center=[0, 0],
            alpha=0.1,
            aa=True,
            r=y_max,
            theta1=115,
            theta2=185,
            fc="cyan",
        )
    )

    # front left
    ax.add_patch(
        mpatches.Wedge(
            center=[0, 0],
            alpha=0.1,
            aa=True,
            r=y_max,
            theta1=25,
            theta2=125,
            fc="cyan",
        )
    )
    # front right
    ax.add_patch(
        mpatches.Wedge(
            center=[0, 0],
            alpha=0.1,
            aa=True,
            r=y_max,
            theta1=35,
            theta2=125,
            fc="cyan",
        )
    )
    # back
    ax.add_patch(
        mpatches.Wedge(
            center=[0, 0],
            alpha=0.1,
            aa=True,
            r=y_max,
            theta1=35,
            theta2=125,
            fc="cyan",
        )
    )
    # back left
    ax.add_patch(
        mpatches.Wedge(
            center=[0, 0],
            alpha=0.1,
            aa=True,
            r=y_max,
            theta1=35,
            theta2=125,
            fc="cyan",
        )
    )
    # back right
    ax.add_patch(
        mpatches.Wedge(
            center=[0, 0],
            alpha=0.1,
            aa=True,
            r=y_max,
            theta1=35,
            theta2=125,
            fc="cyan",
        )
    )

    # Mark ego-vehicle
    ax.arrow(0, 0, 0, 3, color="black", width=0.5, overhang=0.3)
    return fig, ax


def draw_bev_box(
    axis: Axis,
    box: torch.Tesnor,
    color: torch.Tensor,
    history: torch.tensor = torch.empty((0, 7)),
    line_width: int = 2,
):
    """Draws a 3D bounding box in a bird's eye view.

    Args:
        axis (Axis): Matplotlib axis.
        box (torch.Tensor): Bounding box in the format
            [x_c, y_c, z_C, l, w, h, yaw]. Shape [7].
        color (torch.Tensor): Color of the bounding box. Shape [n_boxes, 3]
        history (torch.Tensor): History of the bounding box.
            Shape [n_history, 7]. Defaults to empty tensor.
        line_width (int, optional): Line width of the bounding box.
            Defaults to 2.
    """
    center = np.array(box[:2])
    yaw = box[8]
    l = box[5]
    w = box[4]
    color = tuple((np.array(color) / 255).tolist())

    # Calculate length, width of object
    vec_l = np.array([l * np.cos(yaw), -l * np.sin(yaw)])
    vec_w = np.array(
        [-w * np.cos(yaw - np.pi / 2), w * np.sin(yaw - np.pi / 2)]
    )

    # Make 4 points
    p1 = center + 0.5 * vec_l - 0.5 * vec_w
    p2 = center + 0.5 * vec_l + 0.5 * vec_w
    p3 = center - 0.5 * vec_l + 0.5 * vec_w
    p4 = center - 0.5 * vec_l - 0.5 * vec_w

    # Plot object
    line_style = "-"

    axis.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        line_style,
        c=color,
        linewidth=3 * line_width,
    )
    axis.plot(
        [p1[0], p4[0]],
        [p1[1], p4[1]],
        line_style,
        c=color,
        linewidth=line_width,
    )
    axis.plot(
        [p3[0], p2[0]],
        [p3[1], p2[1]],
        line_style,
        c=color,
        linewidth=line_width,
    )
    axis.plot(
        [p3[0], p4[0]],
        [p3[1], p4[1]],
        line_style,
        c=color,
        linewidth=line_width,
    )

    # Plot center history
    if len(history) > 0:
        yaw_hist = history[:, 8]
        center_hist = history[:, :2]
        for index, ct in enumerate(center_hist):
            yaw = yaw_hist[index].item()
            vec_l = np.array([l * np.cos(yaw), -l * np.sin(yaw)])
            ct_dir = ct + 0.5 * vec_l
            alpha = max(float(index) / len(center_hist), 0.5)
            axis.plot(
                [ct[0], ct_dir[0]],
                [ct[1], ct_dir[1]],
                line_style,
                alpha=alpha,
                c=color,
                linewidth=line_width,
            )
            axis.scatter(
                ct[0],
                ct[1],
                alpha=alpha,
                c=np.array([color]),
                linewidth=line_width,
            )


def draw_bev(
    boxes3d: Tensor, history: list[Tensor] | None = None
) -> np.ndarray:
    """Plots a bird's eye view of the scene with the given bounding boxes.

    Args:
        boxes3d (Tensor): Bounding boxes in the format
            [x_c, y_c, z_c, l, w, h, yaw]. Shaped [n_boxes, 7].
        history (list[Tensor]): History of the bounding boxes.
            Shape [n_boxes, n_history, 7]. Defaults to None.

    Returns:
        np.ndarray: Numpy image rendered top down.
    """
    fig, ax = draw_bev_canvas()

    assert (
        history is None or history.shape[0] == boxes3d.shape[1]
    ), "History and boxes3d must have the same length."

    for idx, box in enumerate(boxes3d):
        hist = history[idx] if history is not None else torch.empty((0, 7))
        draw_bev_box(ax, box, color=torch.tensor([255, 0, 0]), history=hist)

    fig.canvas.figure.tight_layout()
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()

    # canvas.tostring_argb give pixmap in ARGB mode.
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    plt.close(fig)

    buf.shape = (h, w, 4)  # last dim: (alpha, r, g, b)

    # Take only RGB
    buf = buf[:, :, 1:]
    return buf


def draw_lines_match(
    img1: Image.Image,
    img2: Image.Image,
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    radius: int = 5,
) -> Image:  # pragma: no cover
    """Draws matched points between two images.

    This function is used to draw the matches between two images. It is used
    to visualize the matches between the keypoints of two images. The keypoints
    are represented by circles and the lines connecting the circles represent
    the matches between the keypoints.

    Args:
        img1 (Image.Image): First image.
        img2 (Image.Image): Second image.
        pts1 (torch.Tensor): Keypoints of the first image. Shaped [n_pts, 2].
        pts2 (torch.Tensor): Keypoints of the second image. Shaped [n_pts, 2].
        radius (int): Radius of the circles representing the keypoints.

    Returns:
        Image: Image with the keypoints and matches drawn.
    """
    img1 = np.array(img1)
    img2 = np.array(img2)
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    interval = 5

    out = 255 * np.ones(
        (max([rows1, rows2]), cols1 + cols2 + interval, 3), dtype="uint8"
    )
    out[:rows2, cols1 + interval : cols1 + cols2 + interval, :] = img2
    pts2[:, 0] += cols1 + interval

    # Place the first image to the left
    out[:rows1, :cols1, :] = img1

    out_im = Image.fromarray(out)
    draw = ImageDraw.Draw(out_im)

    for pt1, pt2 in zip(pts1, pts2):
        draw.ellipse(
            [tuple(pt1.astype(int) - radius), tuple(pt1.astype(int) + radius)],
            outline=(255, 0, 0),
        )
        draw.ellipse(
            [tuple(pt2.astype(int) - radius), tuple(pt2.astype(int) + radius)],
            outline=(255, 0, 0),
        )
        draw.line(
            [tuple(pt1.astype(int)), tuple(pt2.astype(int))],
            fill=(0, 255, 0),
            width=1,
        )
    return out_im


def imshow_pointcloud(
    points: torch.Tensor,
    image: ImageType,
    camera_intrinsics: Tensor,
    dot_size: int = 3,
    mode: str = "RGB",
) -> None:  # pragma: no cover
    """Show image with pointcloud points.

    Args:
        points (torch.Tensor): Pointcloud points. Shaped [n_pts, 3].
        image (ImageType): Image.
        camera_intrinsics (Tensor): Camera intrinsics.
        dot_size (int, optional): Size of the points. Defaults to 3.
        mode (str, optional): Image mode. Defaults to "RGB".
    """
    image_p = preprocess_image(image, mode)

    pts_2d = project_points(points, camera_intrinsics).round()
    depths = points[:, 2]
    mask = points_inside_image(
        pts_2d, depths, (image_p.shape[1], image_p.shape[0])
    )
    pts_2d = pts_2d[mask].int().cpu().numpy()
    depths = depths[mask].cpu().numpy()

    plt.figure(figsize=(9, 16))
    plt.imshow(image_p)
    plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c=depths, s=dot_size)
    plt.axis("off")
    plt.show()


def bbox3d_to_lines_plotly(
    box3d_corners: NDArrayF64,
) -> tuple[
    list[NDArrayF64], list[NDArrayF64], list[NDArrayF64]
]:  # pragma: no cover
    """Convert 3D boxes to lines.

    Args:
        box3d_corners (NDArrayF64): 3D boxes. Shaped [n_boxes, 8, 3].

    Returns:
        tuple[list[NDArrayF64], list[NDArrayF64], list[NDArrayF64]]: 3D lines.
    """
    ixs_box_0 = [0, 1, 2, 3, 0]
    ixs_box_1 = [4, 5, 6, 7, 4]

    x_lines = [box3d_corners[ixs_box_0, 0]]
    y_lines = [box3d_corners[ixs_box_0, 1]]
    z_lines = [box3d_corners[ixs_box_0, 2]]

    def f_lines_add_nones() -> None:
        """Add nones for lines."""
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    f_lines_add_nones()

    x_lines.extend(box3d_corners[ixs_box_1, 0])
    y_lines.extend(box3d_corners[ixs_box_1, 1])
    z_lines.extend(box3d_corners[ixs_box_1, 2])
    f_lines_add_nones()

    for i in range(4):
        x_lines.extend(box3d_corners[[ixs_box_0[i], ixs_box_1[i]], 0])
        y_lines.extend(box3d_corners[[ixs_box_0[i], ixs_box_1[i]], 1])
        z_lines.extend(box3d_corners[[ixs_box_0[i], ixs_box_1[i]], 2])
        f_lines_add_nones()

    # heading
    x_lines.extend(box3d_corners[[0, 5], 0])
    y_lines.extend(box3d_corners[[0, 5], 1])
    z_lines.extend(box3d_corners[[0, 5], 2])
    f_lines_add_nones()

    x_lines.extend(box3d_corners[[1, 4], 0])
    y_lines.extend(box3d_corners[[1, 4], 1])
    z_lines.extend(box3d_corners[[1, 4], 2])
    f_lines_add_nones()
    return x_lines, y_lines, z_lines


def show_pointcloud(
    points: torch.Tensor,
    colors: torch.Tensor = None,
    axis_mode: AxisMode = AxisMode.OPENCV,
    boxes3d: Tensor | None = None,
    thickness: int = 2,
) -> None:  # pragma: no cover
    """Show pointcloud points using plotly as backend.

    Args:
        points (torch.Tensor): Pointcloud points. Shaped [n_pts, 3].
        colors (AxisMode, optional): RGB color.
        axis_mode (AxisMode, optional): Axis mode. Defaults to AxisMode.OPENCV.
        boxes3d (Tensor, optional): 3D boxes. Shaped [n_boxes, 7].
            Defaults to None.
        thickness (int, optional): Thickness of the points. Defaults to 2.

    Raises:
        ValueError: If axis_mode is not AxisMode.ROS or AxisMode.OPENCV.
    """
    assert (
        PLOTLY_AVAILABLE
    ), "Visualize pointcloud in 3D needs Plotly installed!."
    assert DASH_AVAILABLE, "Visualize pointcloud in 3D needs Dash installed!."
    points = points[:, :3].cpu()

    if colors is None:
        marker = dict(
            color=np.linalg.norm(points, axis=1),
            colorscale="Viridis",
            size=thickness,
        )
    else:
        marker = dict(
            color=colors.cpu().numpy(),
            size=thickness,
        )

    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=marker,
    )

    data = [scatter]
    if boxes3d is not None:
        boxes_corners = boxes3d_to_corners(boxes3d, axis_mode)
        for idx, box in enumerate(boxes_corners):
            color = DEFAULT_COLOR_MAPPING[idx % len(DEFAULT_COLOR_MAPPING)]
            x_lines, y_lines, z_lines = bbox3d_to_lines_plotly(np.array(box))
            lines = go.Scatter3d(
                x=x_lines,
                y=y_lines,
                z=z_lines,
                mode="lines",
                name="lines",
                marker=dict(size=thickness, color=f"rgb{color}"),
            )
            data.append(lines)

    fig = go.Figure(data=data)

    # set to camera appropriate to coordinate system
    if axis_mode == AxisMode.OPENCV:
        camera = dict(
            up=dict(x=0, y=-1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.0, y=0.0, z=-1.25),
        )
    elif axis_mode == AxisMode.ROS:
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.0, y=1.25, z=0),
        )
    else:
        raise ValueError(f"Axis mode {axis_mode} not known.")

    fig.update_layout(scene_camera=camera, scene_aspectmode="data")

    def dash_app() -> None:
        """Establish dash app server."""
        app = dash.Dash(__name__)
        app.layout = html.Div(
            [
                html.Div([dcc.Graph(id="visualization", figure=fig)]),
            ]
        )
        app.run_server(debug=False, port=8080, host="0.0.0.0")

    p = Process(target=dash_app)
    p.start()
    input("Press Enter to continue...")  # pylint: disable=bad-builtin
    p.terminate()
