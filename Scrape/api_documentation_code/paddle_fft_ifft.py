import numpy as np
import paddle

x = np.exp(3j * np.pi * np.arange(7) / 7)
xp = paddle.to_tensor(x)
ifft_xp = paddle.fft.ifft(xp).numpy()
print(ifft_xp)
#  [0.14285714+1.79137191e-01j 0.14285714+6.87963741e-02j
#   0.14285714+1.26882631e-16j 0.14285714-6.87963741e-02j
#   0.14285714-1.79137191e-01j 0.14285714-6.25898038e-01j
#   0.14285714+6.25898038e-01j]