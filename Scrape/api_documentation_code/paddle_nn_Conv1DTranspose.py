import paddle
from paddle.nn import Conv1DTranspose

# shape: (1, 2, 4)
x = paddle.to_tensor([[[4, 0, 9, 7],
                    [8, 0, 9, 2]]], dtype="float32")
# shape: (2, 1, 2)
w = paddle.to_tensor([[[7, 0]],
                    [[4, 2]]], dtype="float32")

conv = Conv1DTranspose(2, 1, 2)
conv.weight.set_value(w)
y = conv(x)
print(y)
# Tensor(shape=[1, 1, 5], dtype=float32, place=Place(gpu:0), stop_gradient=False,
#        [[[60., 16., 99., 75., 4. ]]])