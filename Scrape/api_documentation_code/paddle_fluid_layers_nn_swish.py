# imperative mode
import numpy as np
from paddle import fluid
import paddle.fluid.dygraph as dg

data = np.random.randn(2, 3).astype("float32")
place = fluid.CPUPlace()
with dg.guard(place) as g:
    x = dg.to_variable(data)
    y = fluid.layers.swish(x)
    y_np = y.numpy()
data
# array([[-0.0816701 ,  1.1603649 , -0.88325626],
#        [ 0.7522361 ,  1.0978601 ,  0.12987892]], dtype=float32)
y_np
# array([[-0.03916847,  0.8835007 , -0.25835553],
#        [ 0.51126915,  0.82324016,  0.06915068]], dtype=float32)