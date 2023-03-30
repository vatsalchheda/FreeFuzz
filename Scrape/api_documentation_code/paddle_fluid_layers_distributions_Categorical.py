import numpy as np
from paddle.fluid import layers
from paddle.fluid.layers import Categorical

a_logits_npdata = np.array([-0.602,-0.602], dtype="float32")
a_logits_tensor = layers.create_tensor(dtype="float32")
layers.assign(a_logits_npdata, a_logits_tensor)

b_logits_npdata = np.array([-0.102,-0.112], dtype="float32")
b_logits_tensor = layers.create_tensor(dtype="float32")
layers.assign(b_logits_npdata, b_logits_tensor)

a = Categorical(a_logits_tensor)
b = Categorical(b_logits_tensor)

a.entropy()
# [0.6931472] with shape: [1]

b.entropy()
# [0.6931347] with shape: [1]

a.kl_divergence(b)
# [1.2516975e-05] with shape: [1]