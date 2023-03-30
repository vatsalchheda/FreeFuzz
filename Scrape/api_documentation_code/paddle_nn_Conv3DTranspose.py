import paddle
import paddle.nn as nn

paddle.disable_static()

x_var = paddle.uniform((2, 4, 8, 8, 8), dtype='float32', min=-1., max=1.)

conv = nn.Conv3DTranspose(4, 6, (3, 3, 3))
y_var = conv(x_var)
y_np = y_var.numpy()
print(y_np.shape)
# (2, 6, 10, 10, 10)