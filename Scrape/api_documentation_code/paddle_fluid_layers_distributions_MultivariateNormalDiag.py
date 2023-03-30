import numpy as np
from paddle.fluid import layers
from paddle.fluid.layers import MultivariateNormalDiag

a_loc_npdata = np.array([0.3,0.5],dtype="float32")
a_loc_tensor = layers.create_tensor(dtype="float32")
layers.assign(a_loc_npdata, a_loc_tensor)


a_scale_npdata = np.array([[0.4,0],[0,0.5]],dtype="float32")
a_scale_tensor = layers.create_tensor(dtype="float32")
layers.assign(a_scale_npdata, a_scale_tensor)

b_loc_npdata = np.array([0.2,0.4],dtype="float32")
b_loc_tensor = layers.create_tensor(dtype="float32")
layers.assign(b_loc_npdata, b_loc_tensor)

b_scale_npdata = np.array([[0.3,0],[0,0.4]],dtype="float32")
b_scale_tensor = layers.create_tensor(dtype="float32")
layers.assign(b_scale_npdata, b_scale_tensor)

a = MultivariateNormalDiag(a_loc_tensor, a_scale_tensor)
b = MultivariateNormalDiag(b_loc_tensor, b_scale_tensor)

a.entropy()
# [2.033158] with shape: [1]
b.entropy()
# [1.7777451] with shape: [1]

a.kl_divergence(b)
# [0.06542051] with shape: [1]