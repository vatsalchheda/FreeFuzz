# imperative mode
import numpy as np
from paddle import fluid
import paddle.fluid.dygraph as dg

data = np.random.randn(2, 3).astype("float32")
place = fluid.CPUPlace()
with dg.guard(place) as g:
    x = dg.to_variable(data)
    y = fluid.layers.thresholded_relu(x, threshold=0.1)
    y_np = y.numpy()
data
# array([[ 0.21134382, -1.1805999 ,  0.32876605],
#        [-1.2210793 , -0.7365624 ,  1.0013918 ]], dtype=float32)
y_np
# array([[ 0.21134382, -0.        ,  0.32876605],
#        [-0.        , -0.        ,  1.0013918 ]], dtype=float32)