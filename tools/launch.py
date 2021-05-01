"""Training and prediction command line tool."""

from detectron2.engine import launch

from openmt import config
from openmt.common.utils import default_argument_parser
from openmt.engine import predict, train

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = config.parse_config(args)

    if args.action == "train":
        main_func = train
    elif args.action == "predict":
        main_func = predict
    else:
        raise NotImplementedError(
            f"Detect action {args.action} not " f"implemented!"
        )
    launch(
        main_func,
        cfg.launch.num_gpus,
        num_machines=cfg.launch.num_machines,
        machine_rank=cfg.launch.machine_rank,
        dist_url=cfg.launch.dist_url,
        args=(cfg,),
    )
