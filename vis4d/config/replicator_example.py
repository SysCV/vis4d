"""Tests for replicator  #TODO move to docs, jupyter."""

# from ml_collections import ConfigDict

# from vis4d.config.replicator import (
#     iterable_sampler,
#     linspace_sampler,
#     logspace_sampler,
#     replicate_config,
# )

# config = ConfigDict({"trainer": {"lr": 0.2, "bs": 2}})
# replicated_config = replicate_config(
#     config,
#     sampling_args={"trainer.lr": linspace_sampler(0.01, 0.1, 3)},
#     method="grid",
# )
# for c in replicated_config:
#     print(c)


# print("=" * 20)

# replicated_config = replicate_config(
#     config,
#     sampling_args={
#         "trainer.bs": linspace_sampler(8, 24, 3),
#         "trainer.lr": logspace_sampler(-4, -3, 5),
#     },
#     method="grid",
# )

# for c in replicated_config:
#     print(f"lr {round(c.trainer.lr,3):<7} bs {c.trainer.bs}")


# config.optimizer_name = ""

# config = ConfigDict(
#     {"trainer": {"lr": 0.2, "bs": 2, "optimizer_name": "adam"}}
# )
# replicated_config = replicate_config(
#     config,
#     sampling_args={
#         "trainer.optimizer_name": iterable_sampler(("sgd", "adam")),
#         "trainer.lr": logspace_sampler(-4, -3, 2),
#         "trainer.bs": linspace_sampler(8, 16, 2),
#     },
#     method="grid",
# )

# print("=" * 20)
# for c in replicated_config:
#     print(
#         f"lr {round(c.trainer.lr,3):<7} bs {c.trainer.bs:<4} optim {c.trainer.optimizer_name}"
#     )

# replicated_config = replicate_config(
#     config,
#     sampling_args={
#         "trainer.optimizer_name": iterable_sampler(("sgd", "adam", "test")),
#         "trainer.lr": logspace_sampler(-4, -3, 3),
#         "trainer.bs": linspace_sampler(8, 16, 2),
#     },
#     method="linear",
# )
# print("=" * 20)
# for c in replicated_config:
#     print(
#         f"lr {round(c.trainer.lr,3):<7} bs {c.trainer.bs:<4} optim {c.trainer.optimizer_name}"
#     )
