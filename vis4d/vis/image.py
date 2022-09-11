"""Vis4D Visualization tools for analysis and debugging."""
from multiprocessing import Process
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import patches as mpatches
from matplotlib.ticker import MultipleLocator
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from vis4d.common_to_clean.geometry.projection import (
    generate_depth_map,
    generate_projected_point_mask,
    project_points,
)
from vis4d.struct import Extrinsics, Intrinsics, NDArrayF64, NDArrayUI8

try:  # pragma: no cover
    import dash
    import plotly.graph_objects as go
    from dash import dcc, html

    DASH_INSTALLED = True
except (ImportError, NameError):
    DASH_INSTALLED = False

from .utils import (
    Box3DType,
    ImageType,
    box3d_to_corners,
    preprocess_boxes,
    preprocess_image,
    preprocess_intrinsics,
    preprocess_masks,
)


def imshow(
    image: Union[Image.Image, ImageType], mode: str = "RGB"
) -> None:  # pragma: no cover
    """Imshow method.

    Args:
        image: PIL Image or ImageType (i.e. numpy array, torch.Tensor)
        mode: Image channel format, will be used to convert ImageType to
        an RGB PIL Image.
    """
    if not isinstance(image, Image.Image):
        image = preprocess_image(image, mode)
    plt.imshow(np.asarray(image))
    plt.show()


def imshow_bboxes(
    image: ImageType,
    boxes: Tensor,
    scores: Optional[Tensor] = None,
    class_ids: Optional[Tensor] = None,
    track_ids: Optional[Tensor] = None,
    mode: str = "RGB",
) -> None:  # pragma: no cover
    """Show image with bounding boxes."""
    image = preprocess_image(image, mode)
    box_list, color_list, label_list = preprocess_boxes(
        boxes, scores, class_ids, track_ids
    )
    for box, col, label in zip(box_list, color_list, label_list):
        draw_bbox(image, box, col, label)

    return np.asarray(image)
    # imshow(image)


def imshow_bboxes3d(
    image: ImageType,
    boxes: Box3DType,  # TODO rewrite
    intrinsics: Union[NDArrayF64, Intrinsics],
    mode: str = "RGB",
) -> None:  # pragma: no cover
    """Show image with bounding boxes."""
    image = preprocess_image(image, mode)
    box_list, color_list, label_list = preprocess_boxes(boxes)
    intrinsic_matrix = preprocess_intrinsics(intrinsics)

    for box, col, label in zip(box_list, color_list, label_list):
        draw_bbox3d(image, box, intrinsic_matrix, col, label)

    imshow(image)


def imshow_masks(
    image: ImageType,
    masks: Tensor,
    scores: Optional[Tensor] = None,
    class_ids: Optional[Tensor] = None,
    track_ids: Optional[Tensor] = None,
    mode: str = "RGB",
) -> None:  # pragma: no cover
    """Show image with masks."""
    image = preprocess_image(image, mode)
    mask_list, color_list = preprocess_masks(
        masks, scores, class_ids, track_ids
    )
    for mask, col in zip(mask_list, color_list):
        draw_mask(image, mask, col)

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
        ax.add_patch(
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


def draw_bev(boxes3d: Box3DType, history: List[Box3DType]) -> np.ndarray:
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


def draw_image(
    frame: Union[ImageType, Image.Image],
    boxes2d=None,  # TODO update
    boxes3d: Optional[Box3DType] = None,
    intrinsics: Optional[Union[NDArrayF64, Intrinsics]] = None,
    mode: str = "RGB",
) -> Image.Image:
    """Draw boxes2d on an image."""
    image = (
        preprocess_image(frame, mode)
        if not isinstance(frame, Image.Image)
        else frame
    )
    if boxes2d is not None:
        box_list, col_list, label_list = preprocess_boxes(boxes2d)
        for box, col, label in zip(box_list, col_list, label_list):
            draw_bbox(image, box, col, label)
    if boxes3d is not None:
        assert intrinsics is not None, "Drawing 3D boxes requires intrinsics!"
        intr_matrix = preprocess_intrinsics(intrinsics)
        box_list, col_list, label_list = preprocess_boxes(boxes3d)
        for box, col, label in zip(box_list, col_list, label_list):
            draw_bbox3d(image, box, intr_matrix, col, label)
    return image


def draw_bbox(
    image: Image.Image,
    box: List[float],
    color: Tuple[int],
    label: Optional[str] = None,
) -> None:
    """Draw 2D box onto image."""
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=color)
    if label is not None:
        font = ImageFont.load_default()
        draw.text(box[:2], label, (255, 255, 255), font=font)


def draw_bbox3d(
    image: Image.Image,
    box3d: List[float],
    intrinsics: NDArrayF64,
    color: Tuple[int],
    label: Optional[str] = None,
    camera_near_clip: float = 0.15,
) -> None:  # pragma: no cover
    """Draw 3D box onto image."""
    draw = ImageDraw.Draw(image)
    corners = box3d_to_corners(box3d)
    corners_proj = corners / corners[:, 2:3]
    corners_proj = np.dot(corners_proj, intrinsics.T)

    def draw_line(
        point1: NDArrayF64, point2: NDArrayF64, color: Tuple[int]
    ) -> None:
        if point1[2] < camera_near_clip and point2[2] < camera_near_clip:
            return
        if point1[2] < camera_near_clip:
            point1 = get_intersection_point(point1, point2, camera_near_clip)
        elif point2[2] < camera_near_clip:
            point2 = get_intersection_point(point1, point2, camera_near_clip)
        draw.line((tuple(point1[:2]), tuple(point2[:2])), fill=color)

    def draw_rect(selected_corners: NDArrayF64) -> None:
        prev = selected_corners[-1]
        for corner in selected_corners:
            draw_line(prev, corner, color)
            prev = corner

    # Draw the sides
    for i in range(4):
        draw_line(corners_proj[i], corners_proj[i + 4], color)

    # Draw bottom (first 4 corners) and top (last 4 corners)
    draw_rect(corners_proj[:4])
    draw_rect(corners_proj[4:])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners_proj[:2], axis=0)
    center_bottom = np.mean(corners_proj[:4], axis=0)
    draw_line(center_bottom, center_bottom_forward, color)

    if label is not None:
        font = ImageFont.load_default()
        center_top_forward = tuple(np.mean(corners_proj[2:4], axis=0)[:2])
        draw.text(center_top_forward, label, (255, 255, 255), font=font)


def draw_mask(
    image: Image.Image, mask: NDArrayUI8, color: Tuple[int]
) -> None:  # pragma: no cover
    """Draw mask onto image."""
    draw = ImageDraw.Draw(image)
    # create overlay mask
    mask = np.repeat(mask[:, :, None], 4, axis=2)
    mask[:, :, -1][mask[:, :, -1] == 255] = 128
    draw.bitmap([0, 0], Image.fromarray(mask, mode="RGBA"), fill=color)


def get_intersection_point(
    point1: NDArrayF64, point2: NDArrayF64, camera_near_clip: float
) -> NDArrayF64:  # pragma: no cover
    """Get point intersecting with camera near plane on line point1 -> point2.

    The line is defined by two points (3 dimensional) in camera coordinates.
    """
    cam_dir: NDArrayF64 = np.array([0, 0, 1], dtype=np.float64)
    center_pt: NDArrayF64 = cam_dir * camera_near_clip

    c1, c2, c3 = center_pt
    a1, a2, a3 = cam_dir
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    k_up = abs(a1 * (x1 - c1) + a2 * (y1 - c2) + a3 * (z1 - c3))
    k_down = abs(a1 * (x1 - x2) + a2 * (y1 - y2) + a3 * (z1 - z2))
    if k_up > k_down:
        k = 1
    else:
        k = k_up / k_down
    return (1 - k) * point1 + k * point2


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
    key_extrinsics: Extrinsics,
    key_intrinsics: Intrinsics,
    ref_image: ImageType,
    ref_extrinsics: Extrinsics,
    ref_intrinsics: Intrinsics,
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
    mask = generate_projected_point_mask(
        points_key[:, 2], key_pix, key_im.szie[0], key_im.size[1]
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
    camera_intrinsics: Intrinsics,
    boxes3d: Optional[Box3DType] = None,
    dot_size: int = 3,
    mode: str = "RGB",
) -> None:  # pragma: no cover
    """Show image with pointcloud points."""
    image_p = preprocess_image(image, mode)
    _, pts2d, coloring, _ = generate_depth_map(
        points[:, :3], camera_intrinsics, image_p.size[0], image_p.size[1]
    )
    pts2d, coloring = pts2d.cpu().numpy(), coloring.cpu().numpy()

    plt.figure(figsize=(16, 9))
    plt.scatter(pts2d[:, 0], pts2d[:, 1], c=coloring, s=dot_size)

    if boxes3d is not None:
        imshow_bboxes3d(image, boxes3d, camera_intrinsics)
    else:
        imshow(image)


def plotly_draw_bbox3d(
    box: List[float],
) -> Tuple[
    List[NDArrayF64], List[NDArrayF64], List[NDArrayF64]
]:  # pragma: no cover
    """Plot 3D boxes in 3D space."""
    ixs_box_0 = [0, 1, 2, 3, 0]
    ixs_box_1 = [4, 5, 6, 7, 4]
    corners = box3d_to_corners(box)

    x_lines = [corners[ixs_box_0, 0]]
    y_lines = [corners[ixs_box_0, 1]]
    z_lines = [corners[ixs_box_0, 2]]

    def f_lines_add_nones() -> None:
        """Add nones for lines."""
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    f_lines_add_nones()

    x_lines.extend(corners[ixs_box_1, 0])
    y_lines.extend(corners[ixs_box_1, 1])
    z_lines.extend(corners[ixs_box_1, 2])
    f_lines_add_nones()

    for i in range(4):
        x_lines.extend(corners[[ixs_box_0[i], ixs_box_1[i]], 0])
        y_lines.extend(corners[[ixs_box_0[i], ixs_box_1[i]], 1])
        z_lines.extend(corners[[ixs_box_0[i], ixs_box_1[i]], 2])
        f_lines_add_nones()

    # heading
    x_lines.extend(corners[[0, 5], 0])
    y_lines.extend(corners[[0, 5], 1])
    z_lines.extend(corners[[0, 5], 2])
    f_lines_add_nones()

    x_lines.extend(corners[[1, 4], 0])
    y_lines.extend(corners[[1, 4], 1])
    z_lines.extend(corners[[1, 4], 2])
    f_lines_add_nones()
    return x_lines, y_lines, z_lines


def show_pointcloud(
    points: torch.Tensor,
    boxes3d: Optional[Box3DType] = None,
    thickness: int = 2,
) -> None:  # pragma: no cover
    """Show pointcloud points."""
    assert DASH_INSTALLED, "Visualize pointcloud in 3D needs Dash installed!."
    points = points[:, :3].cpu()

    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(size=thickness),
    )

    data = [scatter]
    if boxes3d is not None:
        box_list, col_list, label_list = preprocess_boxes(boxes3d)
        for box, color, _ in zip(box_list, col_list, label_list):
            x_lines, y_lines, z_lines = plotly_draw_bbox3d(box)
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

    # set to OpenCV based camera system
    camera = dict(
        up=dict(x=0, y=-1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.0, y=0.0, z=-1.25),
    )
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
