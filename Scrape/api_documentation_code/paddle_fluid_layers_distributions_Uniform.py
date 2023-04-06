import numpy as np
from paddle.fluid import layers
from paddle.fluid.layers import Uniform

# Without broadcasting, a single uniform distribution [3, 4]:
u1 = Uniform(low=3.0, high=4.0)
# 2 distributions [1, 3], [2, 4]
u2 = Uniform(low=[1.0, 2.0],
              high=[3.0, 4.0])
# 4 distributions
u3 = Uniform(low=[[1.0, 2.0],
          [3.0, 4.0]],
     high=[[1.5, 2.5],
           [3.5, 4.5]])

# With broadcasting:
u4 = Uniform(low=3.0, high=[5.0, 6.0, 7.0])

# Complete example
value_npdata = np.array([0.8], dtype="float32")
value_tensor = layers.create_tensor(dtype="float32")
layers.assign(value_npdata, value_tensor)

uniform = Uniform([0.], [2.])

sample = uniform.sample([2])
# a random tensor created by uniform distribution with shape: [2, 1]
entropy = uniform.entropy()
# [0.6931472] with shape: [1]
lp = uniform.log_prob(value_tensor)
# [-0.6931472] with shape: [1]