import paddle
import paddle.nn as nn

class LeNet(nn.Layer):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2D(
                1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(
                6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2D(2, 2))

        if num_classes > 0:
            self.fc = nn.Sequential(
                nn.Linear(400, 120),
                nn.Linear(120, 84),
                nn.Linear(
                    84, 10))

    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)
        return x

lenet = LeNet()
# m is the instance of nn.Layer, x is the intput of layer, y is the output of layer.
def count_leaky_relu(m, x, y):
    x = x[0]
    nelements = x.numel()
    m.total_ops += int(nelements)

FLOPs = paddle.flops(lenet, [1, 1, 28, 28], custom_ops= {nn.LeakyReLU: count_leaky_relu},
                    print_detail=True)
print(FLOPs)

#+--------------+-----------------+-----------------+--------+--------+
#|  Layer Name  |   Input Shape   |   Output Shape  | Params | Flops  |
#+--------------+-----------------+-----------------+--------+--------+
#|   conv2d_2   |  [1, 1, 28, 28] |  [1, 6, 28, 28] |   60   | 47040  |
#|   re_lu_2    |  [1, 6, 28, 28] |  [1, 6, 28, 28] |   0    |   0    |
#| max_pool2d_2 |  [1, 6, 28, 28] |  [1, 6, 14, 14] |   0    |   0    |
#|   conv2d_3   |  [1, 6, 14, 14] | [1, 16, 10, 10] |  2416  | 241600 |
#|   re_lu_3    | [1, 16, 10, 10] | [1, 16, 10, 10] |   0    |   0    |
#| max_pool2d_3 | [1, 16, 10, 10] |  [1, 16, 5, 5]  |   0    |   0    |
#|   linear_0   |     [1, 400]    |     [1, 120]    | 48120  | 48000  |
#|   linear_1   |     [1, 120]    |     [1, 84]     | 10164  | 10080  |
#|   linear_2   |     [1, 84]     |     [1, 10]     |  850   |  840   |
#+--------------+-----------------+-----------------+--------+--------+
#Total Flops: 347560     Total Params: 61610