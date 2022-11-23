"""Check if optional packages required by some modules are available."""
from functools import lru_cache
from importlib.util import find_spec


@lru_cache()
def package_available(package_name: str) -> bool:
    """Check if a package is available in your environment."""
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


# io
H5PY_AVAILABLE = package_available("h5py")

# vision
MMCV_AVAILABLE = package_available("mmcv") or package_available("mmcv-full")
MMDET_AVAILABLE = package_available("mmdet")
MMSEG_AVAILABLE = package_available("mmseg")
DETECTRON2_AVAILABLE = package_available("detectron2")

# datasets
WAYMO_AVAILABLE = package_available("waymo")
NUSCENES_AVAILABLE = package_available("nuscenes")
SCALABEL_AVAILABLE = package_available("scalabel")
BDD100K_AVAILABLE = package_available("bdd100k")

# visualization
OPENCV_AVAILABLE = package_available("cv2")
DASH_AVAILABLE = package_available("dash")
OPEN3D_AVAILABLE = package_available("open3d")
PLOTLY_AVAILABLE = package_available("plotly")
