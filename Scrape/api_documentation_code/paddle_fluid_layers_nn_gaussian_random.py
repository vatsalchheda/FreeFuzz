# imperative mode
import numpy as np
from paddle import fluid
import paddle.fluid.dygraph as dg

place = fluid.CPUPlace()
with dg.guard(place) as g:
    x = fluid.layers.gaussian_random((2, 4), mean=2., dtype="float32", seed=10)
    x_np = x.numpy()
x_np
# array([[2.3060477 , 2.676496  , 3.9911983 , 0.9990833 ],
#        [2.8675377 , 2.2279181 , 0.79029655, 2.8447366 ]], dtype=float32)