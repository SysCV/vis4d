"""Training and prediction command line tool using detection API."""

import sys

from systm import config, detect
from systm.engine import default_argument_parser, default_setup

if __name__ == "__main__":
    action = sys.argv[1]
    sys.argv.pop(1)
    args = default_argument_parser().parse_args()
    cfg = config.read_config(args.config_file)

    # convert config to detectron2 format
    cfg = config.to_detectron2(cfg)

    # merge config and args.opts
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    if hasattr(detect, action):
        detect.launch_module(getattr(detect, action), args, cfg)
    else:
        raise ValueError(f"Action {action} not supported!")
