import numpy as np
import paddle

x = np.array([3, 1, 2, 2, 3], dtype=float)
scalar_temp = 0.3
n = x.size
rfftfreq_xp = paddle.fft.rfftfreq(n, d=scalar_temp)
print(rfftfreq_xp)

#  Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#           [0.        , 0.66666669, 1.33333337])