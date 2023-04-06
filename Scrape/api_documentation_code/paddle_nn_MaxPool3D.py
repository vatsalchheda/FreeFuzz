import paddle
import paddle.nn as nn

# max pool3d
input = paddle.uniform([1, 2, 3, 32, 32], dtype="float32", min=-1, max=1)
MaxPool3D = nn.MaxPool3D(kernel_size=2,
                       stride=2, padding=0)
output = MaxPool3D(input)
# output.shape [1, 2, 3, 16, 16]

# for return_mask=True
MaxPool3D = nn.MaxPool3D(kernel_size=2, stride=2, padding=0, return_mask=True)
output, max_indices = MaxPool3D(input)
# output.shape [1, 2, 3, 16, 16], max_indices.shape [1, 2, 3, 16, 16],