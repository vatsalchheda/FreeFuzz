import paddle
import paddle.nn.functional as F

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = F.softplus(x)
print(out)
# [0.513015, 0.598139, 0.744397, 0.854355]