import torch
from collections import OrderedDict

# eights = "/home/yinjiang/systm/examples/deepsort_example/checkpoint/original_ckpt.t7"
# state_dict_t7 = torch.load(weights, map_location=torch.device("cpu"))[
#     "net_dict"
# ]
# state_dict_pth = OrderedDict()
# for k, v in state_dict_t7.items():
#     state_dict_pth["feature_net." + k] = v


# torch.save(
#     state_dict_pth,
#     "/home/yinjiang/systm/examples/deepsort_example/checkpoint/original_ckpt.pth",
# )

weights = "/home/yinjiang/systm/examples/deepsort_example/checkpoint/original_ckpt.pth"
state_dict_pth = torch.load(weights, map_location=torch.device("cpu"))
print("end")
