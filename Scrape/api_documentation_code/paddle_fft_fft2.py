import numpy as np
import paddle

x = np.mgrid[:2, :2][1]
xp = paddle.to_tensor(x)
fft2_xp = paddle.fft.fft2(xp).numpy()
print(fft2_xp)
#  [[ 2.+0.j -2.+0.j]
#   [ 0.+0.j  0.+0.j]]