"""Training and prediction command line tool using detection API."""

from systm import config, detect
from systm.util import default_argument_parser, default_setup

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = config.read_config(args.config)

    # convert config to detectron2 format
    detectron2cfg = config.to_detectron2(cfg)

    # merge config and args.opts
    detectron2cfg.merge_from_list(args.opts)
    detectron2cfg.freeze()
    default_setup(detectron2cfg, args)

    if hasattr(detect, args.action):
        getattr(detect, args.action)(args, detectron2cfg)
    else:
        raise ValueError(f"Action {args.action} not supported!")
