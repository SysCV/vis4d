"""Training and prediction command line tool using detection API."""

import sys
from systm import detect
from systm import config

from detectron2.engine import default_argument_parser

if __name__ == "__main__":
    action = sys.argv[1]
    sys.argv.pop(1)
    args = default_argument_parser().parse_args()
    cfg = config.read_config(args.config_file)

    if hasattr(detect, action):
        getattr(detect, action)(args, cfg)
    else:
        raise ValueError(f'Action {action} not supported!')