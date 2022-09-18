"""FCN tests."""
from time import perf_counter
from typing import Optional, Tuple, Union, NamedTuple
import os

import numpy as np
import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms


from ..base.resnet import ResNet
from ..utils import load_model_checkpoint
from .common import ResizeWithPadding, evaluate_sem_seg
from .fcn import FCNForResNet, FCNLoss, FCNOut


REV_KEYS = [
    (r"^backbone\.", "body."),
    (r"^aux_classifier\.", "heads.0."),
    (r"^classifier\.", "heads.1."),
]


class FCNResNetModel(nn.Module):
    def __init__(self, resnet_model: str = "resnet50") -> None:
        """Init."""
        super().__init__()
        self.basemodel = ResNet(
            resnet_model,
            pretrained=True,
            replace_stride_with_dilation=[False, True, True],
        )
        self.fcn = FCNForResNet(
            self.basemodel.out_channels[4:],
            21,
            resize=(500, 500),
        )
        self.loss = FCNLoss(feature_idx=[4, 5])

    def forward(
        self, images: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Union[FCNLoss, FCNOut]:
        features = self.basemodel(images)
        pred = self.fcn(features)
        if targets is not None:
            losses = self.loss(pred.outputs, targets)
            return losses
        return pred


@torch.no_grad()
def validation_loop(model, val_dataloader):
    """validate current model with test dataset."""
    model.eval()
    print("Running validation...")
    preds = []
    targets = []
    for _, data in enumerate(tqdm.tqdm(val_dataloader)):
        image, target = data
        outputs = model(image.to(device))
        pred = outputs.pred.argmax(1)
        preds.extend(
            [
                pred[i].cpu().numpy().astype(np.int64)
                for i in range(pred.shape[0])
            ]
        )
        targets.extend(
            [
                target[i].cpu().numpy().astype(np.int64)
                for i in range(target.shape[0])
            ]
        )
    metrics, _ = evaluate_sem_seg(preds, targets, num_classes=21)
    log_str = "[Validation] "
    for k, v in metrics.items():
        log_str += f"{k}: {v:.4f}, "
    print(log_str.rstrip(", "), flush=True)
    return metrics


def training_loop(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    log_step: int = 5,
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
            losses = model(image.to(device), target.to(device))
            total_loss = losses.total_loss
            total_loss.backward()
            optimizer.step()
            toc = tic = perf_counter()

            # print statistics
            losses = dict(time=toc - tic, loss=total_loss)
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
        metrics = validation_loop(model, val_loader)
        torch.save(
            model.state_dict(),
            f"{ckpt_dir}/epoch_{epoch + 1}_mIoU_{metrics['mIoU']:.2f}.pt",
        )
    print("training done.")


def setup(args):
    # setup model and dataloader
    fcn_resnet = FCNResNetModel(resnet_model=args.resnet_model)
    optimizer = optim.SGD(fcn_resnet.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40, 60, 80], gamma=0.1
    )

    # setup test dataset
    transform = transforms.Compose(
        [
            ResizeWithPadding((500, 500)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    target_transform = transforms.Compose(
        [
            ResizeWithPadding((500, 500)),
            transforms.ToTensor()
            # PascalMaskEncoder()
        ]
    )
    train_loader = DataLoader(
        VOCSegmentation(
            root="vis4d-workspace/data/voc2012",
            year="2012",
            image_set="train",
            transform=transform,
            target_transform=target_transform,
        ),
        batch_size=16,
        shuffle=True,
    )
    val_loader = DataLoader(
        VOCSegmentation(
            root="vis4d-workspace/data/voc2012",
            year="2012",
            image_set="val",
            transform=transform,
            target_transform=target_transform,
        ),
        batch_size=16,
        shuffle=False,
    )
    return fcn_resnet, optimizer, scheduler, train_loader, val_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VOC train/eval.")
    parser.add_argument("--lr", default=1e-2, help="learning rate.")
    parser.add_argument(
        "-c", "--ckpt", default=None, help="path of model to eval."
    )
    parser.add_argument(
        "--save_name",
        default="fcn_resnet50_voc2012",
        help="folder name where models are saved.",
    )
    parser.add_argument(
        "--resnet_model",
        default="resnet50",
        choices=["resnet50", "resnet101"],
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
            num_epochs=100,
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
        validation_loop(fcn_resnet, val_loader)
