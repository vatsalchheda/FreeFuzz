import paddle
import numpy as np

x = np.array([[0, 1, 3, 0],
            [1, 1, 0, 1]])
paddle.incubate.asp.calculate_density(x) # 0.625