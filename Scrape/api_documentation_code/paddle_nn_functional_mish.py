import paddle
import paddle.nn.functional as F

x = paddle.to_tensor([-5., 0., 5.])
out = F.mish(x) # [-0.03357624, 0., 4.99955208]