import paddle
import paddle.nn.functional as F

x = paddle.to_tensor([2., 0., 1.])
out = F.thresholded_relu(x)
print(out)
# Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [2., 0., 0.])