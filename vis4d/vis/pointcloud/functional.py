"""Function interface for point cloud visualization functions."""

from __future__ import annotations

from vis4d.common.typing import ArrayLikeFloat, ArrayLikeInt

from ..util import DEFAULT_COLOR_MAPPING
from .scene import Scene3D
from .viewer import Open3DVisualizationBackend, PointCloudVisualizerBackend


def show_3d(
    scene: Scene3D,
    viewer: PointCloudVisualizerBackend = Open3DVisualizationBackend(
        class_color_mapping=DEFAULT_COLOR_MAPPING
    ),
) -> None:
    """Shows a given 3D scene.

    This method shows a 3D visualization of a given 3D scene. Use the viewer
    attribute to use different visualization backends (e.g. open3d)

    Args:
        scene (Scene3D): The 3D scene that should be visualized.
        viewer (PointCloudVisualizerBackend, optional): The Visualization
            backend that should be used to visualize the scene.
            Defaults to Open3DVisualizationBackend.
    """
    viewer.add_scene(scene)
    viewer.show()
    viewer.reset()


def draw_points(
    points_xyz: ArrayLikeFloat,
    colors: ArrayLikeFloat | None = None,
    classes: ArrayLikeInt | None = None,
    instances: ArrayLikeInt | None = None,
    transform: ArrayLikeFloat | None = None,
    scene: Scene3D | None = None,
) -> Scene3D:
    """Adds pointcloud data to a 3D scene for visualization purposes.

    Args:
        points_xyz: xyz coordinates of the points shape [N, 3]
        classes: semantic ids of the points shape [N, 1]
        instances: instance ids of the points shape [N, 1]
        colors: colors of the points shape [N,3] and ranging from  [0,1]
        transform: Optional 4x4 SE3 transform that transforms the point data
            into a static reference frame.
        scene (Scene3D | None): Visualizer that should be used to display the
            data.
    """
    if scene is None:
        scene = Scene3D()

    return scene.add_pointcloud(
        points_xyz, colors, classes, instances, transform
    )


def show_points(
    points_xyz: ArrayLikeFloat,
    colors: ArrayLikeFloat | None = None,
    classes: ArrayLikeInt | None = None,
    instances: ArrayLikeInt | None = None,
    transform: ArrayLikeFloat | None = None,
    viewer: PointCloudVisualizerBackend = Open3DVisualizationBackend(
        class_color_mapping=DEFAULT_COLOR_MAPPING
    ),
) -> None:
    """Visualizes a pointcloud with color and semantic information.

    Args:
        points_xyz: xyz coordinates of the points shape [N, 3]
        classes: semantic ids of the points shape [N, 1]
        instances: instance ids of the points shape [N, 1]
        colors: colors of the points shape [N,3] and ranging from  [0,1]
        transform: Optional 4x4 SE3 transform that transforms the point data
            into a static reference frame
        viewer (PointCloudVisualizerBackend, optional): The Visualization
            backend that should be used to visualize the scene.
            Defaults to Open3DVisualizationBackend.
    """
    show_3d(
        draw_points(points_xyz, colors, classes, instances, transform), viewer
    )
