"""Example to show connected components in the config."""
from vis4d.config.data_graph import prints_datagraph_for_config
from vis4d.config.util import instantiate_classes

from .faster_rcnn.faster_rcnn_coco import get_config

dg = prints_datagraph_for_config(instantiate_classes(get_config()))
print(dg)
