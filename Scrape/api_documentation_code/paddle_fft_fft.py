import numpy as np
import paddle

x = np.exp(3j * np.pi * np.arange(7) / 7)
xp = paddle.to_tensor(x)
fft_xp = paddle.fft.fft(xp).numpy()
print(fft_xp)
#  [1.+1.25396034e+00j 1.+4.38128627e+00j 1.-4.38128627e+00j
#   1.-1.25396034e+00j 1.-4.81574619e-01j 1.+8.88178420e-16j
#   1.+4.81574619e-01j]