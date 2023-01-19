import re

import numpy as np

lines = """
[00:00:18] ·MAIN· dnn.dnn.0.weight torch.Size([1000, 1536])
[00:00:18] ·MAIN· dnn.dnn.0.bias torch.Size([1000])
[00:00:18] ·MAIN· dnn.dnn.3.weight torch.Size([1000, 1000])
[00:00:18] ·MAIN· dnn.dnn.3.bias torch.Size([1000])
[00:00:18] ·MAIN· dnn.dnn.6.weight torch.Size([1000, 1000])
[00:00:18] ·MAIN· dnn.dnn.6.bias torch.Size([1000])
[00:00:18] ·MAIN· cross_net.cross_net.0.bias torch.Size([1536])
[00:00:18] ·MAIN· cross_net.cross_net.0.weight.weight torch.Size([1, 1536])
[00:00:18] ·MAIN· cross_net.cross_net.1.bias torch.Size([1536])
[00:00:18] ·MAIN· cross_net.cross_net.1.weight.weight torch.Size([1, 1536])
[00:00:18] ·MAIN· cross_net.cross_net.2.bias torch.Size([1536])
[00:00:18] ·MAIN· cross_net.cross_net.2.weight.weight torch.Size([1, 1536])
[00:00:18] ·MAIN· prediction.weight torch.Size([1, 2536])
[00:00:18] ·MAIN· prediction.bias torch.Size([1])
"""


lines = lines.split('\n')
pattern = 'torch.Size\((.*?)\)'

total_params = 0


for line in lines:
    result = re.findall(pattern, line)
    if result:
        params = eval(result[0])
        total_params += np.prod(params)

print(f'Total trainable parameters (M): {total_params / 1e6:.2f}')
