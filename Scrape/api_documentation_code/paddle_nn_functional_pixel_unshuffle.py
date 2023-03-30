import paddle
import paddle.nn.functional as F
x = paddle.randn([2, 1, 12, 12])
out = F.pixel_unshuffle(x, 3)
print(out.shape)
# [2, 9, 4, 4]