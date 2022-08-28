Command line arguments overview
=================================

Standard Options
+++++++++++++++++++++++++++++++++++++++
action (positional argument): train / test / predict
config: Filepath to config file

Launch Options
+++++++++++++++++++++++++++++++++++++++
- work_dir: Specific directory to save checkpoints, logs, etc. Integrates with exp_name and version to work_dir/exp_name/version. Default: ./vis4d-workspace/
- exp_name: Name of current experiment. Default: <name of model>
- version: Version of current experiment. Default: <timestamp>
- input_dir: Input directory in case you want to run inference on a folder with input data (e.g. images that can be temporally sorted by name).
- find_unused_parameters: Activates PyTorch checking for unused parameters in DDP setting. Deactivated by default for better performance.
- visualize: If you're running in predict mode, this option lets you visualize the model predictions in the output_dir.
- seed: Set random seed for numpy, torch, python. Default: None, i.e. no specific random seed is chosen.
- weights: Filepath for weights to load in test / predict. Default: "best",  will load the best checkpoint in work_dir/exp_name/version.
- checkpoint_period: After N epochs, save out checkpoints (default: 1)
- resume: Whether to resume from weights (if specified), or last ckpt in work_dir/exp_name/version.
- pin_memory: Enable/Disable pin_memory option for dataloader workers in
- training.
- wandb: Use weights and biases logging instead of tensorboard (default).
- strict: Whether to enforce keys in weights to be consistent with model's.
- tqdm: Activate tqdm based terminal logging behavior.

For the PyTorchLightning Trainer Options, see `here <https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html>`_

Basic command line usage
================================

Enumerate possible launch options
To list all possible arguments and options, use:

.. code:: bash

    python3 -m vis4d.engine.trainer -h

Training a model
Example using the provided config

.. code:: bash

    python3 -m vis4d.engine.trainer train --config configs/experiments/qdtrack/qdtrack_R_50_FPN_bdd100k.toml --gpus 8


Set --gpus to <num_gpus> if training on GPU or optionally, select the specific GPUs with <GPU_id0>,<GPU_id1>,... (see below)

While there are multiple distributed backends available, it is strongly discouraged to use dp in PyTorchLightning, as well as using ddp_spawn together with multiple dataset workers.

Training a model with additional arguments or overriding arguments in config file
Example using the provided config

.. code:: bash

    python3 -m vis4d.engine.trainer train --config configs/experiments/qdtrack/qdtrack_R_50_FPN_bdd100k.toml --gpus 2,3,4,5 --cfg-options model.lr_scheduler.warmup_steps=1000,model.optimizer.lr=0.01


Additional config parameters in format key=value must be separated by commas.

Resume a training

.. code:: bash

    python3 -m vis4d.engine.trainer train --resume --work_dir ./vis4d_workspace/ --exp_name QDTrack --version 2021-08-29_16-31-19


This will resume training at the checkpoint saved in ./vis4d-workspace/QDTrack/2021-08-29_16-31-19/checkpoints/last.ckpt

or resume from a specific checkpoint using:

.. code:: bash

    python3 -m vis4d.engine.trainer train --resume --weights /path/to/resume.ckpt


Initialize from checkpoint

.. code:: bash

    python3 -m vis4d.engine.trainer train --config configs/experiments/qdtrack/qdtrack_R_50_FPN_bdd100k.toml --weights /path/to/checkpoint.ckpt


This will start a new training initialized from the given checkpoint.

Testing a model

.. code:: bash

    python3 -m vis4d.engine.trainer test --config configs/experiments/qdtrack/qdtrack_R_50_FPN_bdd100k.toml --weights /path/to/model.ckpt


This will test the model on the given test dataset and output the corresponding metrics as well as predictions in output_dir

Run inference on the test dataset(s)

.. code:: bash

    python3 -m vis4d.engine.trainer predict --config configs/experiments/qdtrack/qdtrack_R_50_FPN_bdd100k.toml --weights /path/to/model.ckpt


This will run inference on the test datasets specified in the config file and save the predictions to the working directory.

Run inference on a list of images

.. code:: bash

    python3 -m vis4d.engine.trainer predict --config configs/experiments/qdtrack/qdtrack_R_50_FPN_bdd100k.toml --input_dir /path/to/folder --weights /path/to/model.ckpt


The folder should contain either subdirectories with images or images directly to run inference on. The images will be sorted by name, which is important to note in case you run a tracking model. The sorted images are treated as a video sequence.

Benchmarking

.. code:: bash

    python3 -m vis4d.engine.trainer train --config configs/experiments/qdtrack/qdtrack_R_50_FPN_bdd100k.toml --max_epochs 1 --limit_train_batches 15 --limit_val_batches 15 --profiler simple


You can profile your implementation easily by setting the profiler option for the PyTorchLightning Trainer, along with some specifications on how you want your benchmark run to be executed (e.g. limit steps to 15 for train / val).
See the `here <https://pytorch-lightning.readthedocs.io/en/latest/advanced/profiler.html>`_ for more details.

Config structure
===================

In the config, you define the important parts of your pipeline. The config is managed as pydantic BaseModel, and can be loaded from .toml or .yaml files.
The important options are:

- model: Defines the parameters for your model. The parameters are dependent on your model implementation. Each model config can specify its own attributes. Each model config inherits from `vis4d.op.BaseModel`.
- train: List of Datasets used for training, e.g. Scalabel, BDD100K, COCO, MOTChallenge, etc. Defined in `vis4d.data.datasets.BaseDatasetLoader`
- train_handler: DatasetHandler for training. Defines behavior of the training data loader if one global configuration of all datasets is desired. Includes augmentations and data postprocessing. Defined in `vis4d.data.Vis4DDatasetHandler`.
- test: List of Datasets used for testing
- launch: Launch configuration. These arguments are identical to the Vis4D command line arguments, which can be specified here in the config or in the command line itself. Defined in `vis4d.config.Launch`.
- trainer: Optionally, the config can contain arguments for the PyTorchLightning trainer. This can be used to specify the run configuration, e.g. evaluation period, total number of steps / epochs, logging period, etc.

Note: Arguments specified in trainer will overwrite command line arguments. E.g. max_epochs specified in the config file cannot be overwritten by --max_epochs X


Dataset Structure
===================
TODO explain how datasets work in Vis4D
Explain mapping from category strings to model output
