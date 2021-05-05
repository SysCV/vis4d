"""Load MOTChallenge data and convert to scalabel format."""
from typing import Dict, List, Optional

from scalabel.label.typing import Frame


def convert_and_load_motchallenge(
    json_path: str,
    image_root: str,
    dataset_name: Optional[str] = None,
    ignore_categories: Optional[List[str]] = None,
    name_mapping: Optional[Dict[str, str]] = None,
    prepare_frames: bool = True,
) -> List[Frame]:
    """Convert motchallenge annotations to scalabel format and prepare them."""
    raise NotImplementedError
