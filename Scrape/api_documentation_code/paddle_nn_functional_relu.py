import paddle
import paddle.nn.functional as F

x = paddle.to_tensor([-2, 0, 1], dtype='float32')
out = F.relu(x)
print(out)
# [0., 0., 1.]