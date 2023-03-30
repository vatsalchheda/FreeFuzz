import paddle
from paddle.distribution import Normal

# Define a single scalar Normal distribution.
dist = Normal(loc=0., scale=3.)
# Define a batch of two scalar valued Normals.
# The first has mean 1 and standard deviation 11, the second 2 and 22.
dist = Normal(loc=[1., 2.], scale=[11., 22.])
# Get 3 samples, returning a 3 x 2 tensor.
dist.sample([3])

# Define a batch of two scalar valued Normals.
# Both have mean 1, but different standard deviations.
dist = Normal(loc=1., scale=[11., 22.])

# Complete example
value_tensor = paddle.to_tensor([0.8], dtype="float32")

normal_a = Normal([0.], [1.])
normal_b = Normal([0.5], [2.])
sample = normal_a.sample([2])
# a random tensor created by normal distribution with shape: [2, 1]
entropy = normal_a.entropy()
# [1.4189385] with shape: [1]
lp = normal_a.log_prob(value_tensor)
# [-1.2389386] with shape: [1]
p = normal_a.probs(value_tensor)
# [0.28969154] with shape: [1]
kl = normal_a.kl_divergence(normal_b)
# [0.34939718] with shape: [1]