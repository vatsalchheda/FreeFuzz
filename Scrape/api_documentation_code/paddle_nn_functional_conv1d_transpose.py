import paddle
import paddle.nn.functional as F

# shape: (1, 2, 4)
x = paddle.to_tensor([[[4, 0, 9, 7],
                      [8, 0, 9, 2,]]], dtype="float32")
# shape: (2, 1, 2)
w = paddle.to_tensor([[[7, 0]],
                      [[4, 2]]], dtype="float32")

y = F.conv1d_transpose(x, w)
print(y)
# Tensor(shape=[1, 1, 5], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[60., 16., 99., 75., 4. ]]])