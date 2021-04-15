"""Training and prediction command line tool using tracking API."""

from openmt import config, track
from openmt.common.utils import default_argument_parser

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = config.parse_config(args)

    if args.action == "train":
        track.train(cfg)
    elif args.action == "predict":
        track.predict(cfg)
