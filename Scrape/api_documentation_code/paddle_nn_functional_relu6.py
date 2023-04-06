import paddle
import paddle.nn.functional as F

x = paddle.to_tensor([-1, 0.3, 6.5])
out = F.relu6(x)
print(out)
# [0, 0.3, 6]