import paddle
import paddle.nn.functional as F

x = paddle.randn(shape=[2,9,4,4])
out_var = F.pixel_shuffle(x, 3)
out = out_var.numpy()
print(out.shape)
# (2, 1, 12, 12)