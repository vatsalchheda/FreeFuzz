import paddle
import paddle.nn as nn

# max pool2d
input = paddle.uniform([1, 3, 32, 32], dtype="float32", min=-1, max=1)
MaxPool2D = nn.MaxPool2D(kernel_size=2,
                       stride=2, padding=0)
output = MaxPool2D(input)
# output.shape [1, 3, 16, 16]

# for return_mask=True
MaxPool2D = nn.MaxPool2D(kernel_size=2, stride=2, padding=0, return_mask=True)
output, max_indices = MaxPool2D(input)
# output.shape [1, 3, 16, 16], max_indices.shape [1, 3, 16, 16],