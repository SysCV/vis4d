"""Vis4D Visualization tools for analysis and debugging."""
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash
from multiprocessing import Process

from vis4d.struct import Intrinsics, Extrinsics, NDArrayF64, NDArrayUI8
from vis4d.common.geometry.projection import project_points

from .utils import (
    Box3DType,
    BoxType,
    ImageType,
    InsMaskType,
    SemMaskType,
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
    image: ImageType, boxes: BoxType, mode: str = "RGB"
) -> None:  # pragma: no cover
    """Show image with bounding boxes."""
    image = preprocess_image(image, mode)
    box_list, color_list, label_list = preprocess_boxes(boxes)
    for box, col, label in zip(box_list, color_list, label_list):
        draw_bbox(image, box, col, label)

    imshow(image)


def imshow_bboxes3d(
    image: ImageType,
    boxes: Box3DType,
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
    image: ImageType, masks: Union[InsMaskType, SemMaskType], mode: str = "RGB"
) -> None:  # pragma: no cover
    """Show image with masks."""
    image = preprocess_image(image, mode)
    mask_list, color_list = preprocess_masks(masks)
    for mask, col in zip(mask_list, color_list):
        draw_mask(image, mask, col)

    imshow(image)


def draw_image(
    frame: Union[ImageType, Image.Image],
    boxes2d: Optional[BoxType] = None,
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
    corners_proj = np.dot(corners_proj, intrinsics.T)  # type: ignore

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
    cam_dir = np.array([0, 0, 1])
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
    return (1 - k) * point1 + k * point2  # type: ignore


def drawlinesMatch(img1, img2, pts1, pts2, concat_row=True, radius=5):
    img1 = np.array(img1)
    img2 = np.array(img2)
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    interval = 5
    if concat_row:
        out = 255 * np.ones(
            (max([rows1, rows2]), cols1 + cols2 + interval, 3), dtype="uint8"
        )
        out[:rows2, cols1 + interval : cols1 + cols2 + interval, :] = img2
        pts2[:, 0] += cols1 + interval
    else:
        out = 255 * np.ones(
            (rows1 + rows2 + interval, max(cols1, cols2), 3), dtype="uint8"
        )
        out[rows1 + interval : rows1 + rows2 + interval, :cols2] = img2
        pts2[:, 1] += rows1 + interval

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
    key_points: torch.tensor,
    key_points_extrinsics: Extrinsics,
):
    key_im, ref_im = preprocess_image(key_image), preprocess_image(ref_image)

    hom_points = torch.cat(
        [key_points[:, :3], torch.ones_like(key_points[:, 0:1])], -1
    )
    points_world = hom_points @ key_points_extrinsics.transpose().tensor[0]

    points_key = (
        points_world @ key_extrinsics.inverse().transpose().tensor[0]
    )[:, :3]
    points_ref = (
        points_world @ ref_extrinsics.inverse().transpose().tensor[0]
    )[:, :3]
    key_pix = project_points(points_key, key_intrinsics).cpu().numpy()
    ref_pix = project_points(points_ref, ref_intrinsics).cpu().numpy()

    mask = np.ones(key_pix.shape[0], dtype=bool)
    mask = np.logical_and(mask, points_key.cpu().numpy()[:, -1] > 0)
    mask = np.logical_and(mask, points_ref.cpu().numpy()[:, -1] > 0)
    mask = np.logical_and(mask, key_pix[:, 0] > 0)
    mask = np.logical_and(mask, key_pix[:, 0] < key_im.size[0] - 1)
    mask = np.logical_and(mask, key_pix[:, 1] > 0)
    mask = np.logical_and(mask, key_pix[:, 1] < key_im.size[1] - 1)
    mask = np.logical_and(mask, ref_pix[:, 0] > 0)
    mask = np.logical_and(mask, ref_pix[:, 0] < ref_im.size[0] - 1)
    mask = np.logical_and(mask, ref_pix[:, 1] > 0)
    mask = np.logical_and(mask, ref_pix[:, 1] < ref_im.size[1] - 1)
    ref_pix = ref_pix[mask]
    key_pix = key_pix[mask]

    perm = torch.randperm(key_pix.shape[0])[:10]
    key_pix = key_pix[perm]
    ref_pix = ref_pix[perm]

    corresp_im = drawlinesMatch(key_im, ref_im, key_pix, ref_pix)
    imshow(corresp_im)


def imshow_lidar(
    points: torch.tensor,
    points_extrinsics: Extrinsics,
    image: ImageType,
    camera_extrinsics: Extrinsics,
    camera_intrinsics: Union[NDArrayF64, Intrinsics],
    boxes: Union[BoxType, Box3DType],
    box_mode: str = "3D",
    dot_size: int = 3,
    mode: str = "RGB",
):
    """Show image with lidar points."""
    center = torch.cat([points[:, :3], torch.ones_like(points[:, 0:1])], -1)

    points_world = center @ points_extrinsics.transpose().tensor[0][:, :3]

    center = torch.cat([points_world, torch.ones_like(points[:, 0:1])], -1)

    points_cam = (
        center @ camera_extrinsics.inverse().transpose().tensor[0][:, :3]
    )

    pts2d = project_points(points_cam, camera_intrinsics).cpu().numpy()

    image_p = preprocess_image(image, mode)

    depths = points_cam[:, 2].cpu().numpy()
    coloring = depths

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1.0)
    mask = np.logical_and(mask, pts2d[:, 0] > 1)
    mask = np.logical_and(mask, pts2d[:, 0] < image_p.size[0] - 1)
    mask = np.logical_and(mask, pts2d[:, 1] > 1)
    mask = np.logical_and(mask, pts2d[:, 1] < image_p.size[1] - 1)

    pts2d = pts2d[mask, :]
    coloring = coloring[mask]

    plt.figure(figsize=(16, 9))
    plt.scatter(pts2d[:, 0], pts2d[:, 1], c=coloring, s=dot_size)

    if box_mode == "3D":
        imshow_bboxes3d(image, boxes, camera_intrinsics)
    elif box_mode == "2D":
        imshow_bboxes(image, boxes)


def plotly_draw_bbox3d(box: List[float]):
    x_lines = []
    y_lines = []
    z_lines = []

    def f_lines_add_nones():
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    ixs_box_0 = [0, 1, 2, 3, 0]
    ixs_box_1 = [4, 5, 6, 7, 4]
    corners = box3d_to_corners(box)

    x_lines.extend(corners[ixs_box_0, 0])
    y_lines.extend(corners[ixs_box_0, 1])
    z_lines.extend(corners[ixs_box_0, 2])
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
    points: torch.tensor,
    points_extrinsics: Extrinsics,
    camera_extrinsics: Extrinsics,
    boxes3d: Box3DType = None,
    thickness=2,
):
    """Show lidar points."""
    points = torch.cat([points[:, :3], torch.ones_like(points[:, 0:1])], -1)
    points_world = points @ points_extrinsics.transpose().tensor[0]
    points = (
        points_world @ camera_extrinsics.inverse().transpose().tensor[0]
    )[:, :3]
    points = points.cpu()

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
        for box, color, label in zip(box_list, col_list, label_list):
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

    def dash_app():
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
