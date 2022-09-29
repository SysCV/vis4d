"""FCN tests."""
import os
from time import perf_counter
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation

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
from vis4d.op.segment.testcase.utils import collate_fn
from vis4d.op.utils import load_model_checkpoint

REV_KEYS = [
    (r"^backbone\.", "body."),
    (r"^aux_classifier\.", "heads.0."),
    (r"^classifier\.", "heads.1."),
]


@torch.no_grad()
def validation_loop(
    model: nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    epoch: int,
    visualization_idx: List[int] = [0],
    visualization_outdir: str = "",
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
                pred_list, f"{visualization_outdir}/epoch={epoch}"
            )
            if epoch == 0:
                save_output_images(target_list, f"{visualization_outdir}/gt")

    metrics, _ = evaluate_sem_seg(preds, targets, num_classes=21)
    metrics["mIoU (ignored BG)"] = float(np.nanmean(metrics["IoUs"][1:]))
    metrics["Acc (ignored BG)"] = float(np.nanmean(metrics["Accs"][1:]))
    log_str = f"[{epoch}, Validation] "
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
    num_epochs: int = 10,
    log_step: int = 5,
    val_step: int = 5,
    ckpt_dir: str = "vis4d-workspace/test/fcnresnet50_voc2012",
):
    """Training loop."""
    print("Start training...", flush=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    running_losses = {}
    for epoch in range(num_epochs):
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
                log_str = f"[{epoch + 1}, {i + 1:5d} / {len(train_loader)}] "
                for k, v in running_losses.items():
                    log_str += f"{k}: {v / log_step:.4f}, "
                print(log_str.rstrip(", "), flush=True)
                running_losses = {}

        scheduler.step()
        if epoch % val_step == 0:
            metrics = validation_loop(
                model,
                val_loader,
                epoch,
                visualization_outdir=f"{ckpt_dir}/pred",
            )
            torch.save(
                model.state_dict(),
                f"{ckpt_dir}/epoch_{epoch + 1}_mIoU_{metrics['mIoU']:.2f}.pt",
            )
    print("training done.")


def visualize_prediction(args):
    # setup model and dataloader
    pred_dir = f"vis4d-workspace/test/{args.save_name}/pred"
    val_loader = DataLoader(
        VOCSegmentation(
            root="vis4d-workspace/data/voc2012",
            year="2012",
            image_set="val",
            transforms=SegmentationPresetRaw(),
        ),
        batch_size=1,
        shuffle=False,
    )
    visualization_idx = list(range(8))
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

    pred_list = read_output_images(f"{pred_dir}/epoch={args.epochs}")
    assert len(pred_list) == len(img_list)
    img_list = blend_images(img_list, pred_list)
    save_output_images(
        img_list, f"{pred_dir}/epoch={args.epoch}_with_img", colorize=False
    )


def setup(args):
    # setup model and dataloader
    fcn_resnet = FCN_ResNet(base_model=args.base_model)
    if args.optim == "SGD":
        optimizer = optim.SGD(
            fcn_resnet.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )
    elif args.optim == "Adam":
        optimizer = optim.Adam(
            fcn_resnet.parameters(), lr=args.lr, weight_decay=5e-4
        )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40, 60, 80], gamma=0.2
    )

    train_loader = DataLoader(
        VOCSegmentation(
            root="vis4d-workspace/data/voc2012",
            year="2012",
            image_set="train",
            transforms=SegmentationPresetTrain(base_size=500, crop_size=512),
        ),
        collate_fn=collate_fn,
        batch_size=8,
        shuffle=True,
    )
    val_loader = DataLoader(
        VOCSegmentation(
            root="vis4d-workspace/data/voc2012",
            year="2012",
            image_set="val",
            transforms=SegmentationPresetEval(base_size=512),
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
        "--lr", default=5e-6, type=float, help="learning rate."
    )
    parser.add_argument(
        "-c", "--ckpt", default=None, help="path of model to eval."
    )
    parser.add_argument(
        "-e", "--epochs", type=int, help="number of epochs to train."
    )
    parser.add_argument("--optim", default="SGD", help="optimizer")
    parser.add_argument(
        "--save_name",
        default="fcn_resnet50_voc2012",
        help="folder name where models are saved.",
    )
    parser.add_argument(
        "--base_model",
        default="resnet50",
        choices=["resnet50", "resnet101", "vgg13", "vgg16"],
        help="select the ResNet model used for base model.",
    )
    parser.add_argument("-n", "--num_gpus", default=1, help="number of gpus")
    args = parser.parse_args()

    device = torch.device("cuda")
    fcn_resnet, optimizer, scheduler, train_loader, val_loader = setup(args)
    fcn_resnet.to(device)
    if args.ckpt is None:
        if args.num_gpus > 1:
            fcn_resnet = nn.DataParallel(
                fcn_resnet, device_ids=[device, torch.device("cuda:1")]
            )
        ckpt_dir = f"vis4d-workspace/test/{args.save_name}"
        training_loop(
            fcn_resnet,
            train_loader,
            val_loader,
            num_epochs=args.epochs,
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
            ckpt = torch.load(args.ckpt)
            fcn_resnet.load_state_dict(ckpt)
        validation_loop(fcn_resnet, val_loader, epoch=args.epochs)
        visualize_prediction(args)
