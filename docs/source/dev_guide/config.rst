Config
=====


Customizable configs
-----
Our dataclass configs allow you to easily plug in different permutations of models, dataloaders, modules, etc. and modify all parameters from a typed CLI supported by tyro.

Base components
-----
All basic, reusable config components can be found in nerfstudio/configs/base_config.py. The Config class at the bottom of the file is the upper-most config level and stores all of the sub-configs needed to get started with training.

You can browse this file and read the attribute annotations to see what configs are available and what each specifies.

Creating new configs
-----
If you are interested in creating a brand new model or data format, you will need to create a corresponding config with associated parameters you want to expose as configurable.

Let’s say you want to create a new model called Nerfacto. You can create a new Model class that extends the base class as described here. Before the model definition, you define the actual NerfactoModelConfig which points to the NerfactoModel class (make sure to wrap the _target classes in a field as shown below).
