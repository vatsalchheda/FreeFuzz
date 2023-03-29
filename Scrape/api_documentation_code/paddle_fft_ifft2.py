import numpy as np
import paddle

x = np.mgrid[:2, :2][1]
xp = paddle.to_tensor(x)
ifft2_xp = paddle.fft.ifft2(xp).numpy()
print(ifft2_xp)
#  [[ 0.5+0.j -0.5+0.j]
#   [ 0. +0.j  0. +0.j]]