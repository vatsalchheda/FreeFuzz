import paddle
import paddle.nn as nn

data = paddle.uniform([1, 3, 32], dtype="float32", min=-1, max=1)
MaxPool1D = nn.MaxPool1D(kernel_size=2, stride=2, padding=0)
pool_out = MaxPool1D(data)
# pool_out shape: [1, 3, 16]

MaxPool1D = nn.MaxPool1D(kernel_size=2, stride=2, padding=0, return_mask=True)
pool_out, indices = MaxPool1D(data)
# pool_out shape: [1, 3, 16], indices shape: [1, 3, 16]