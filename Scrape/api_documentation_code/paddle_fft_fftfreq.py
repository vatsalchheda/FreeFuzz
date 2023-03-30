import numpy as np
import paddle

x = np.array([3, 1, 2, 2, 3], dtype=float)
scalar_temp = 0.5
n = x.size
fftfreq_xp = paddle.fft.fftfreq(n, d=scalar_temp)
print(fftfreq_xp)

#  Tensor(shape=[5], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#           [ 0.        ,  0.40000001,  0.80000001, -0.80000001, -0.40000001])