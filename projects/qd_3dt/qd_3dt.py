"""Quasi-dense 3D Tracking model."""
from pytorch_lightning.utilities.cli import instantiate_class
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from vis4d.model import QD3DT
from vis4d.struct import ArgsType


class StandardQD3DT(QD3DT):  # type: ignore
    """Standard Quasi-dense 3D Tracking model."""

    def configure_optimizers(self):
        """Configure optimizers and schedulers of model."""
        new_params = []
        base_lr = self.optimizer_init["init_args"]["lr"]
        for name, param in self.named_parameters():
            param_group = {"params": [param]}
            if not param.requires_grad:
                new_params.append(param_group)
                continue

            bbox_head = [
                "detector.rpn_head.mm_dense_head.rpn_cls.weight",
                "detector.rpn_head.mm_dense_head.rpn_reg.weight",
                "detector.roi_head.mm_roi_head.bbox_head.fc_cls.weight",
                "detector.roi_head.mm_roi_head.bbox_head.fc_reg.weight",
                "bbox_3d_head.dep_convs.0.weight",
                "bbox_3d_head.dep_convs.1.weight",
                "bbox_3d_head.dep_convs.2.weight",
                "bbox_3d_head.dep_convs.3.weight",
                "bbox_3d_head.dim_convs.0.weight",
                "bbox_3d_head.dim_convs.1.weight",
                "bbox_3d_head.dim_convs.2.weight",
                "bbox_3d_head.dim_convs.3.weight",
                "bbox_3d_head.rot_convs.0.weight"
                "bbox_3d_head.rot_convs.1.weight",
                "bbox_3d_head.rot_convs.2.weight",
                "bbox_3d_head.rot_convs.3.weight",
                "bbox_3d_head.fc_dep.weight",
                "bbox_3d_head.fc_dep_uncer.weight",
                "bbox_3d_head.fc_dim.weight",
                "bbox_3d_head.fc_rot.weight",
                "bbox_3d_head.fc_2dc.weight",
            ]

            # Overwrite bbox head lr
            if name in bbox_head:
                param_group["lr"] = base_lr * 10.0
                rank_zero_info(f"{name} with lr_multi: {10.0}")
            new_params.append(param_group)
        optimizer = instantiate_class(new_params, self.optimizer_init)
        scheduler = instantiate_class(optimizer, self.lr_scheduler_init)
        return [optimizer], [scheduler]
