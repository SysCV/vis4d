"""Check if optional packages required byh some modules are available."""
from pytorch_lightning.utilities.imports import _RequirementAvailable

# general
H5PY_AVAILABLE = _RequirementAvailable("h5py")
SCALABEL_AVAILABLE = _RequirementAvailable("scalabel")

# vision
MMCV_AVAILABLE = _RequirementAvailable("mmcv") or _RequirementAvailable(
    "mmcv-full"
)
MMDET_AVAILABLE = _RequirementAvailable("mmdet==2.20")
MMSEG_AVAILABLE = _RequirementAvailable("mmseg")
DETECTRON2_AVAILABLE = _RequirementAvailable("detectron2")

# datasets
WAYMO_AVAILABLE = _RequirementAvailable("waymo")
NUSCENES_AVAILABLE = _RequirementAvailable("nuscenes")

# visualization
OPENCV_AVAILABLE = _RequirementAvailable("cv2")
DASH_AVAILABLE = _RequirementAvailable("dash")
