# Optimizer

To use it with PyTorch Lightning trainer, our optimizer cfg is actually the list of optimizers yet only one optimizer is supported now.

Within the config, optimizer is written as follow:

```python
optimizer=class_config(
    SGD, lr=params.lr, momentum=0.9, weight_decay=0.0001
),
```

## Learning rate schedular

One optimizer can be worked with multiple learning rate schedulars.

Learning rate schedular can be divided into `epoch_based` or `step_based`.

With different `begin` and `end` to control which interval will the schedular be effective.

```python
lr_schedulers=[
    get_lr_scheduler_cfg(
        class_config(
            LinearLR, start_factor=0.001, total_iters=500
        ),
        end=500,
        epoch_based=False,
    ),
    get_lr_scheduler_cfg(
        class_config(MultiStepLR, milestones=[8, 11], gamma=0.1),
    ),
]
```

As shown above, this optimzer has two schedular.

The first one `LinearLR` will end at 500 steps (since `epoch_based=False`) and each iteration/step it will step and change the learning rate, i.e. linear learning rate warmup.

The second one `MultiStepLR` will step at epoch 8 and 11.
