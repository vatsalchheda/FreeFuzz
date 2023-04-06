import numpy as np
import paddle

x = np.mgrid[:5, :5][0].astype(np.float64)
xp = paddle.to_tensor(x)
ihfft2_xp = paddle.fft.ihfft2(xp).numpy()
print(ihfft2_xp)
# [[ 2. +0.j          0. +0.j          0. +0.j        ]
#  [-0.5-0.68819096j  0. +0.j          0. +0.j        ]
#  [-0.5-0.16245985j  0. +0.j          0. +0.j        ]
#  [-0.5+0.16245985j  0. +0.j          0. +0.j        ]
#  [-0.5+0.68819096j  0. +0.j          0. +0.j        ]]