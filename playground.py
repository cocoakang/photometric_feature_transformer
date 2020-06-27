import torch
import math
import numpy as np
import matplotlib.pyplot as plt

torch.random.manual_seed(666)

wo_dot_n = torch.rand(13,3)
print(wo_dot_n)
tmp_visible_flag = wo_dot_n > 0.5
visible_num = torch.sum(torch.where(tmp_visible_flag,torch.ones_like(wo_dot_n),torch.zeros_like(wo_dot_n)),dim=1)
print(visible_num)
invalid_idxes = torch.where(visible_num < 1)[0]
print(invalid_idxes)
invalid_num = invalid_idxes.size()[0]
print(invalid_num)

new_positions = torch.from_numpy(np.random.rand(invalid_num,3).astype(np.float32))
print(new_positions)

wo_dot_n[invalid_idxes] = new_positions
print(wo_dot_n)