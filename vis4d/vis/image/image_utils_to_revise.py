"""Vis4D Visualization tools for analysis and debugging."""
from __future__ import annotations

from multiprocessing import Process
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches as mpatches
from matplotlib.ticker import MultipleLocator
from PIL import Image, ImageDraw
from torch import Tensor

from vis4d.common import NDArrayF64, NDArrayUI8
from vis4d.common.imports import DASH_AVAILABLE, PLOTLY_AVAILABLE
from vis4d.data.const import AxisMode
from vis4d.op.box.box3d import boxes3d_to_corners
from vis4d.op.geometry.projection import (
    generate_depth_map,
    points_inside_image,
    project_points,
)
from vis4d.vis.util import DEFAULT_COLOR_MAPPING

ImageType = Union[torch.Tensor, NDArrayUI8, NDArrayF64]

ColorType = Union[
    Union[Tuple[int], str],
    List[Union[Tuple[int], str]],
    List[List[Union[Tuple[int], str]]],
]

if DASH_AVAILABLE and PLOTLY_AVAILABLE:
    import dash
    import plotly.graph_objects as go
    from dash import dcc, html


COLOR_PALETTE = DEFAULT_COLOR_MAPPING
NUM_COLORS = 50


def preprocess_boxes(
    boxes: Tensor,
    scores: Optional[Tensor] = None,
    class_ids: Optional[Tensor] = None,
    track_ids: Optional[Tensor] = None,
    color_idx: int = 0,
) -> tuple[list[list[float]], list[tuple[int]], list[str]]:
    """Preprocess BoxType to boxes / colors / labels for drawing."""
    boxes_list = boxes.detach().cpu().numpy().tolist()

    if scores is not None:
        scores = scores.detach().cpu().numpy().tolist()
    else:
        scores = [None for _ in range(len(boxes_list))]

    if track_ids is not None:
        track_ids = track_ids.detach().cpu().numpy()
        if len(track_ids.shape) > 1:
            track_ids = track_ids.squeeze(-1)
    else:
        track_ids = [None for _ in range(len(boxes_list))]

    if class_ids is not None:
        class_ids = class_ids.detach().cpu().numpy()
    else:
        class_ids = [None for _ in range(len(boxes_list))]

    labels, draw_colors = [], []
    for s, t, c in zip(scores, track_ids, class_ids):
        if t is not None:
            draw_color = COLOR_PALETTE[int(t) % NUM_COLORS]
        elif c is not None:
            draw_color = COLOR_PALETTE[int(c) % NUM_COLORS]
        else:
            draw_color = COLOR_PALETTE[color_idx % NUM_COLORS]

        label = ""
        if t is not None:
            label += str(int(t))
        if c is not None:
            label += "," + str(int(c))

        if s is not None:
            label += f",{s * 100:.1f}%"
        labels.append(label)
        draw_colors.append(draw_color)

    return boxes_list, draw_colors, labels


def preprocess_masks(
    masks: Tensor,
    scores: Optional[Tensor] = None,
    class_ids: Optional[Tensor] = None,
    track_ids: Optional[Tensor] = None,
    color_idx: int = 0,
) -> tuple[list[NDArrayUI8], list[tuple[int]]]:
    """Preprocess masks for drawing."""
    if isinstance(masks, list):
        result_mask, result_color = [], []
        for i, m in enumerate(masks):
            mask, color = preprocess_masks(m, i)  # type: ignore
            result_mask.extend(mask)
            result_color.extend(color)
        return result_mask, result_color

    if masks.dim() == 2:
        class_ids = torch.unique(masks)
        masks_list = np.stack(
            [
                ((masks == i).cpu().numpy() * 255).astype(np.uint8)
                for i in class_ids
            ]
        )
    else:
        masks_list = (masks.cpu().numpy() * 255).astype(np.uint8)

    if track_ids is not None:
        track_ids = track_ids.cpu().numpy()
        if len(track_ids.shape) > 1:
            track_ids = track_ids.squeeze(-1)
    else:
        track_ids = [None for _ in range(len(masks_list))]

    if class_ids is not None:
        class_ids = class_ids.cpu().numpy()
    else:
        class_ids = [None for _ in range(len(masks_list))]

    draw_colors = []
    for t, c in zip(track_ids, class_ids):
        if t is not None:
            draw_color = COLOR_PALETTE[int(t) % NUM_COLORS]
        elif c is not None:
            draw_color = COLOR_PALETTE[int(c) % NUM_COLORS]
        else:
            draw_color = COLOR_PALETTE[color_idx % NUM_COLORS]
        draw_colors.append(draw_color)

    return masks_list, draw_colors


def preprocess_image(image: ImageType, mode: str = "RGB") -> Image.Image:
    """Validate and convert input image.

    Args:
        image: CHW or HWC image (ImageType) with C = 3.
        mode: input channel format (e.g. BGR, HSV). More info
        at https://pillow.readthedocs.io/en/stable/handbook/concepts.html

    Returns:
        PIL.Image.Image: Processed image in RGB.
    """
    assert len(image.shape) == 3
    assert image.shape[0] == 3 or image.shape[-1] == 3

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if not image.shape[-1] == 3:
        image = image.transpose(1, 2, 0)
    min_val, max_val = (np.min(image, axis=(0, 1)), np.max(image, axis=(0, 1)))

    image = image.astype(np.float32)

    image = (image - min_val) / (max_val - min_val) * 255.0

    if mode == "BGR":
        image = image[..., [2, 1, 0]]
        mode = "RGB"

    return Image.fromarray(image.astype(np.uint8), mode=mode).convert("RGB")


def imshow_bboxes3d(
    image: ImageType,
    boxes: Tensor,
    intrinsics: Tensor,
    mode: str = "RGB",
) -> None:  # pragma: no cover
    """Show image with bounding boxes."""
    image = preprocess_image(image, mode)
    boxes_corners = boxes3d_to_corners(boxes, AxisMode.OPENCV)
    box_list, color_list, label_list = preprocess_boxes(boxes_corners)
    intrinsics = intrinsics.detach().cpu().numpy()

    for box, col, label in zip(box_list, color_list, label_list):
        draw_bbox3d(image, np.array(box), intrinsics, col, label)

    imshow(image)


def draw_bev_canvas(
    x_min: int = -55,
    x_max: int = 55,
    y_min: int = -55,
    y_max: int = 55,
    fig_size: int = 10,
    dpi: int = 100,
    interval: int = 10,
):
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


def draw_bev_box(ax, box, color, label, hist, line_width: int = 2):
    center = np.array(box[:2])
    yaw = box[8]
    l = box[5]
    w = box[4]
    color = tuple((np.array(color) / 255).tolist())

    # Calculate length, width of object
    vec_l = [l * np.cos(yaw), -l * np.sin(yaw)]
    vec_w = [-w * np.cos(yaw - np.pi / 2), w * np.sin(yaw - np.pi / 2)]
    vec_l = np.array(vec_l)
    vec_w = np.array(vec_w)

    # Make 4 points
    p1 = center + 0.5 * vec_l - 0.5 * vec_w
    p2 = center + 0.5 * vec_l + 0.5 * vec_w
    p3 = center - 0.5 * vec_l + 0.5 * vec_w
    p4 = center - 0.5 * vec_l - 0.5 * vec_w

    # Plot object
    line_style = "-"
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        line_style,
        c=color,
        linewidth=3 * line_width,
    )
    ax.plot(
        [p1[0], p4[0]],
        [p1[1], p4[1]],
        line_style,
        c=color,
        linewidth=line_width,
    )
    ax.plot(
        [p3[0], p2[0]],
        [p3[1], p2[1]],
        line_style,
        c=color,
        linewidth=line_width,
    )
    ax.plot(
        [p3[0], p4[0]],
        [p3[1], p4[1]],
        line_style,
        c=color,
        linewidth=line_width,
    )

    # Plot center history
    if len(hist) > 0:
        yaw_hist = hist[:, 8]
        center_hist = hist[:, :2]
        for index, ct in enumerate(center_hist):
            yaw = yaw_hist[index].item()
            vec_l = np.array([l * np.cos(yaw), -l * np.sin(yaw)])
            ct_dir = ct + 0.5 * vec_l
            alpha = max(float(index) / len(center_hist), 0.5)
            ax.plot(
                [ct[0], ct_dir[0]],
                [ct[1], ct_dir[1]],
                line_style,
                alpha=alpha,
                c=color,
                linewidth=line_width,
            )
            ax.scatter(
                ct[0],
                ct[1],
                alpha=alpha,
                c=np.array([color]),
                linewidth=line_width,
            )


def draw_bev(boxes3d: Tensor, history: list[Tensor]) -> np.ndarray:
    fig, ax = draw_bev_canvas()

    box_list, col_list, label_list = preprocess_boxes(boxes3d)
    for box, col, label in zip(box_list, col_list, label_list):
        track_id = int(label.split(",")[0])
        hist = []
        for hist_boxes3d in history:
            for box3d in hist_boxes3d:
                if box3d.track_ids[0] == track_id:
                    hist.append(box3d.boxes[0])
        if len(hist) > 0:
            hist = torch.stack(hist, 0)
        draw_bev_box(ax, box, col, label, hist)

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
    """Draw matched lines."""
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


def imshow_correspondence(
    key_image: ImageType,
    key_extrinsics: Tensor,
    key_intrinsics: Tensor,
    ref_image: ImageType,
    ref_extrinsics: Tensor,
    ref_intrinsics: Tensor,
    key_points: torch.Tensor,
) -> None:  # pragma: no cover
    """Draw corresponded pointcloud points."""
    key_im, ref_im = preprocess_image(key_image), preprocess_image(ref_image)

    hom_points = torch.cat(
        [key_points[:, :3], torch.ones_like(key_points[:, 0:1])], -1
    )

    points_key = key_points[:, :3]
    points_ref = (
        hom_points
        @ key_extrinsics.transpose().tensor[0]
        @ ref_extrinsics.inverse().transpose().tensor[0]
    )[:, :3]
    key_pix = project_points(points_key, key_intrinsics)
    mask = points_inside_image(
        key_pix, points_key[:, 2], (key_im.size[0], key_im.size[1])
    )

    _, ref_pix, _, mask = generate_depth_map(
        points_ref, ref_intrinsics, ref_im.size[0], ref_im.size[1], mask
    )
    key_pix = key_pix[mask]

    perm = torch.randperm(key_pix.shape[0])[:10]
    key_pix = key_pix[perm].cpu().numpy()
    ref_pix = ref_pix[perm].cpu().numpy()

    corresp_im = draw_lines_match(key_im, ref_im, key_pix, ref_pix)
    imshow(corresp_im)


def imshow_pointcloud(
    points: torch.Tensor,
    image: ImageType,
    camera_intrinsics: Tensor,
    boxes3d: Tensor | None = None,
    dot_size: int = 3,
    mode: str = "RGB",
) -> None:  # pragma: no cover
    """Show image with pointcloud points."""
    image_p = preprocess_image(image, mode)

    pts_2d = project_points(points, camera_intrinsics).round()
    depths = points[:, 2]
    mask = points_inside_image(
        pts_2d, depths, (image_p.size[1], image_p.size[0])
    )
    pts_2d = pts_2d[mask].int().cpu().numpy()
    depths = depths[mask].cpu().numpy()

    plt.figure(figsize=(9, 16))
    plt.imshow(image_p)
    plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c=depths, s=dot_size)
    plt.axis("off")

    if boxes3d is not None:
        imshow_bboxes3d(image, boxes3d, camera_intrinsics)
    else:
        imshow(image)


def plotly_draw_bbox3d(
    box3d_corners: NDArrayF64,
) -> tuple[
    list[NDArrayF64], list[NDArrayF64], list[NDArrayF64]
]:  # pragma: no cover
    """Plot 3D boxes in 3D space."""
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
    axis_mode: AxisMode = AxisMode.OPENCV,
    boxes3d: Tensor | None = None,
    thickness: int = 2,
) -> None:  # pragma: no cover
    """Show pointcloud points."""
    assert (
        PLOTLY_AVAILABLE
    ), "Visualize pointcloud in 3D needs Plotly installed!."
    assert DASH_AVAILABLE, "Visualize pointcloud in 3D needs Dash installed!."
    points = points[:, :3].cpu()

    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            color=np.linalg.norm(points, axis=1),
            colorscale="Viridis",
            size=thickness,
        ),
    )

    data = [scatter]
    if boxes3d is not None:
        boxes_corners = boxes3d_to_corners(boxes3d, axis_mode)
        box_list, col_list, label_list = preprocess_boxes(boxes_corners)
        for box, color, _ in zip(box_list, col_list, label_list):
            x_lines, y_lines, z_lines = plotly_draw_bbox3d(np.array(box))
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
    input("Press Enter to continue...")
    p.terminate()
