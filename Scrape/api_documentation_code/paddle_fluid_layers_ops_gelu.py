# imperative mode
import numpy as np
from paddle import fluid
import paddle.fluid.dygraph as dg

data = np.random.randn(2, 3).astype("float32")
place = fluid.CPUPlace()
with dg.guard(place) as g:
    x = dg.to_variable(data)
    y = fluid.layers.gelu(x)
    y_np = y.numpy()
data
# array([[ 0.87165993, -1.0541513 , -0.37214822],
#        [ 0.15647964,  0.32496083,  0.33045998]], dtype=float32)
y_np
# array([[ 0.70456535, -0.15380788, -0.13207214],
#        [ 0.08796856,  0.20387867,  0.2080159 ]], dtype=float32)