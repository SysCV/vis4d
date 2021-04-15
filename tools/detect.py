"""Training and prediction command line tool using detection API."""

from openmt import config, detect
from openmt.common.utils import default_argument_parser

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = config.parse_config(args)

    if args.action == "train":
        detect.train(cfg)
    elif args.action == "predict":
        detect.predict(cfg)
