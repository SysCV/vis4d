"""Utils for parallel processing."""
import itertools
from typing import Dict, List

from detectron2.utils import comm
from scalabel.label.typing import Frame


def gather_predictions(
    predictions: Dict[str, List[Frame]]
) -> Dict[str, List[Frame]]:
    """Gather prediction dict in distributed setting."""
    comm.synchronize()
    predictions_list = comm.gather(predictions, dst=0)

    result = {}
    for key in predictions:
        prediction_list = [p[key] for p in predictions_list]
        result[key] = list(itertools.chain(*prediction_list))
    return result
