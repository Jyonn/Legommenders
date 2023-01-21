import re

import numpy as np

lines = """
[00:05:28] ·MAIN· user_encoder.multi_head_attention.in_proj_weight torch.Size([192, 64])
[00:05:28] ·MAIN· user_encoder.multi_head_attention.in_proj_bias torch.Size([192])
[00:05:28] ·MAIN· user_encoder.multi_head_attention.out_proj.weight torch.Size([64, 64])
[00:05:28] ·MAIN· user_encoder.multi_head_attention.out_proj.bias torch.Size([64])
[00:05:28] ·MAIN· user_encoder.linear.weight torch.Size([64, 64])
[00:05:28] ·MAIN· user_encoder.linear.bias torch.Size([64])
[00:05:28] ·MAIN· user_encoder.additive_attention.encoder.0.weight torch.Size([64, 64])
[00:05:28] ·MAIN· user_encoder.additive_attention.encoder.0.bias torch.Size([64])
[00:05:28] ·MAIN· user_encoder.additive_attention.encoder.2.weight torch.Size([1, 64])
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
