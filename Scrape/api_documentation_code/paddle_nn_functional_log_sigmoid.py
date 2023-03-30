import paddle
import paddle.nn.functional as F

x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
out = F.log_sigmoid(x) # [-0.313262 -0.126928 -0.0485874 -0.0181499]