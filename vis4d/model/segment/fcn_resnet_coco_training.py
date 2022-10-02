"""FCN tests."""
import os
from time import perf_counter
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from vis4d.model.segment.common import (
    blend_images,
    evaluate_sem_seg,
    read_output_images,
    save_output_images,
)
from vis4d.model.segment.fcn_resnet import FCN_ResNet
from vis4d.op.segment.testcase.presets import (
    SegmentationPresetEval,
    SegmentationPresetRaw,
    SegmentationPresetTrain,
)
from vis4d.op.segment.testcase.utils import collate_fn, get_coco
from vis4d.op.utils import load_model_checkpoint

REV_KEYS = [
    (r"^backbone\.", "basemodel.body."),
    (r"^aux_classifier\.", "fcn.heads.0."),
    (r"^classifier\.", "fcn.heads.1."),
]


@torch.no_grad()
def validation_loop(
    model: nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    cur_iter: int,
    visualization_idx: List[int] = list(range(8)),
    output_dir: str = "",
) -> Dict[str, Any]:
    """validate current model with test dataset."""
    model.eval()
    print("Running validation...")
    preds = []
    targets = []
    for idx, data in enumerate(tqdm.tqdm(val_dataloader)):
        image, target = data
        out = model(image.to(device))
        pred = out.pred.argmax(1)
        pred_list = [
            pred[i].cpu().numpy().astype(np.int64)
            for i in range(pred.shape[0])
        ]
        target_list = [
            target[i].cpu().numpy().astype(np.int64)
            for i in range(target.shape[0])
        ]
        preds.extend(pred_list)
        targets.extend(target_list)
        if idx in visualization_idx:
            save_output_images(
                pred_list,
                f"{output_dir}/iter={cur_iter}",
                offset=idx * len(pred_list),
            )
            if cur_iter == 0:
                save_output_images(
                    target_list,
                    f"{output_dir}/gt",
                    offset=idx * len(pred_list),
                )

    metrics, _ = evaluate_sem_seg(preds, targets, num_classes=21)
    log_str = f"[{cur_iter}, Validation] "
    for k, v in metrics.items():
        if isinstance(v, float):
            log_str += f"{k}: {v:.4f}, "
        elif isinstance(v, np.ndarray):
            log_str += f"{k}: "
            for vv in v:
                log_str += f"{vv:.1f} "
            log_str += ", "
    print(log_str.rstrip(", "), flush=True)
    return metrics


def training_loop(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    total_iters: int,
    log_step: int = 5,
    val_step: int = 2000,
    ckpt_dir: str = "vis4d-workspace/test/fcn_resnet50_coco2017",
):
    """Training loop."""
    print("Start training...", flush=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    running_losses = {}
    cur_iter = 0
    while cur_iter < total_iters:
        model.train()
        for i, data in enumerate(train_loader):
            image, target = data
            tic = perf_counter()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            target = target.squeeze(1).long()
            out, loss = model(image.to(device), target.to(device))
            loss.total_loss.backward()
            optimizer.step()
            toc = perf_counter()

            # print statistics
            pred = out.pred.argmax(1)
            p_acc = torch.mean((pred.cpu() == target).float())
            mask = target > 0
            p_acc_ig = torch.mean((pred[mask].cpu() == target[mask]).float())
            losses = dict(
                time=toc - tic,
                lr=scheduler.get_last_lr()[0],
                loss=loss.total_loss.item(),
                pAcc=p_acc,
                pAccIg=p_acc_ig,
            )
            for k, v in losses.items():
                if k in running_losses:
                    running_losses[k] += v
                else:
                    running_losses[k] = v
            if i % log_step == (log_step - 1):
                log_str = f"[{i + 1:5d} / {len(train_loader)}] "
                for k, v in running_losses.items():
                    log_str += f"{k}: {v / log_step:.4f}, "
                print(log_str.rstrip(", "), flush=True)
                running_losses = {}

            if cur_iter % val_step == val_step - 1:
                metrics = validation_loop(
                    model,
                    val_loader,
                    cur_iter,
                    output_dir=f"{ckpt_dir}/pred",
                )
                torch.save(
                    model.state_dict(),
                    f"{ckpt_dir}/iter_{cur_iter + 1}_mIoU_{metrics['mIoU']:.2f}_Acc_{metrics['Acc']:.2f}.pt",
                )
                model.train()

            cur_iter += 1
            scheduler.step()
            if cur_iter > total_iters:
                break
    print("training done.")


def visualize_loop(
    pred_dir: str,
    output_dir: str,
    visualization_idx: List[int] = list(range(64)),
) -> None:
    """Visualization via blending predictions with images."""
    # setup model and dataloader
    val_loader = DataLoader(
        get_coco(
            root="vis4d-workspace/data/coco",
            image_set="val",
            transforms=SegmentationPresetRaw(),
        ),
        batch_size=1,
        shuffle=False,
    )
    img_list = []
    for idx, data in enumerate(tqdm.tqdm(val_loader)):
        if idx in visualization_idx:
            image, _ = data
            img_list.extend(
                [
                    (image[i].cpu().numpy() * 255).astype(np.uint8)
                    for i in range(image.shape[0])
                ]
            )
        if idx > max(visualization_idx):
            break

    pred_list = read_output_images(pred_dir)
    assert len(pred_list) == len(img_list)
    img_list = blend_images(img_list, pred_list)
    save_output_images(img_list, output_dir, colorize=False)


def setup(args):
    # setup model and dataloader
    fcn_resnet = FCN_ResNet(base_model=args.base_model, resize=(520, 520))
    if args.optim == "SGD":
        optimizer = optim.SGD(
            fcn_resnet.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True,
        )
    elif args.optim == "Adam":
        optimizer = optim.Adam(
            fcn_resnet.parameters(), lr=args.lr, weight_decay=1e-4
        )
    lr_scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer, args.total_iters, power=0.9
    )

    if args.lr_warmup_iters > 0:
        warmup_iters = args.lr_warmup_iters
        warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=warmup_iters,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, lr_scheduler],
            milestones=[warmup_iters],
        )
    else:
        scheduler = lr_scheduler

    train_loader = DataLoader(
        get_coco(
            root="vis4d-workspace/data/coco",
            image_set="train",
            transforms=SegmentationPresetTrain(base_size=520, crop_size=520),
        ),
        collate_fn=collate_fn,
        batch_size=8,
        shuffle=True,
    )
    val_loader = DataLoader(
        get_coco(
            root="vis4d-workspace/data/coco",
            image_set="val",
            transforms=SegmentationPresetEval(base_size=520),
        ),
        collate_fn=collate_fn,
        batch_size=8,
        shuffle=False,
    )
    return fcn_resnet, optimizer, scheduler, train_loader, val_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VOC train/eval.")
    parser.add_argument(
        "--lr", default=1e-2, type=float, help="learning rate."
    )
    parser.add_argument(
        "-c", "--ckpt", default=None, help="path of model to eval."
    )
    parser.add_argument(
        "--total_iters",
        type=int,
        default=50000,
        help="number of epochs to train.",
    )
    parser.add_argument("--optim", default="SGD", help="optimizer")
    parser.add_argument(
        "--save_name",
        default="fcn_resnet50_coco2017",
        help="folder name where models are saved.",
    )
    parser.add_argument(
        "--base_model",
        default="resnet50",
        choices=["resnet50", "resnet101", "vgg13", "vgg16"],
        help="select the ResNet model used for base model.",
    )
    parser.add_argument(
        "--lr_warmup_iters", default=1000, type=int, help="LR warmup iters."
    )
    parser.add_argument(
        "-n", "--num_gpus", default=1, type=int, help="number of gpus"
    )
    args = parser.parse_args()

    ckpt_dir = f"vis4d-workspace/test/{args.save_name}"
    device = torch.device("cuda")
    fcn_resnet, optimizer, scheduler, train_loader, val_loader = setup(args)
    fcn_resnet.to(device)
    if args.ckpt is None:
        if args.num_gpus > 1:
            print("GPUs:", torch.cuda.device_count())
            fcn_resnet = nn.DataParallel(fcn_resnet)
        training_loop(
            fcn_resnet,
            train_loader,
            val_loader,
            total_iters=args.total_iters,
            ckpt_dir=ckpt_dir,
        )
    else:
        if args.ckpt == "torchvision":
            weights = (
                "https://download.pytorch.org/models/"
                "fcn_resnet50_coco-1167a1af.pth"
            )
            load_model_checkpoint(fcn_resnet, weights, REV_KEYS)
        else:
            ckpt_path = f"{ckpt_dir}/{args.ckpt}"
            ckpt = torch.load(ckpt_path)
            print(f"Loaded checkpoint from {ckpt_path}.")
            fcn_resnet.load_state_dict(ckpt)
        validation_loop(
            fcn_resnet,
            val_loader,
            cur_iter=0,
            output_dir=f"{ckpt_dir}/pred_{args.ckpt}",
        )
        visualize_loop(
            f"{ckpt_dir}/pred_{args.ckpt}/iter=0",
            f"{ckpt_dir}/pred_{args.ckpt}/iter=0_with_gt",
        )
