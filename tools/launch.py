"""Training, testing and prediction command line tool."""
from vist import config
from vist.common.utils import default_argument_parser
from vist.engine import predict, test, train

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = config.parse_config(args)

    if args.action == "train":
        main_func = train
    elif args.action == "test":
        main_func = test
    elif args.action == "predict":
        main_func = predict  # type: ignore
    else:
        raise NotImplementedError(f"Action {args.action} not implemented!")
    main_func(cfg)
