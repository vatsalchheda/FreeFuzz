import numpy as np
import paddle

x = np.array([3, 1, 2, 2, 3], dtype=float)
n = x.size
fftfreq_xp = paddle.fft.fftfreq(n, d=0.3)
res = paddle.fft.ifftshift(fftfreq_xp).numpy()
print(res)
#  [ 1.3333334 -1.3333334 -0.6666667  0.         0.6666667]