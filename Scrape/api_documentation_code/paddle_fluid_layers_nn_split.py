import paddle
paddle.enable_static()
import paddle.fluid as fluid

# input is a Tensor which shape is [3, 9, 5]
input = fluid.data(
     name="input", shape=[3, 9, 5], dtype="float32")

out0, out1, out2 = fluid.layers.split(input, num_or_sections=3, dim=1)
# out0.shape [3, 3, 5]
# out1.shape [3, 3, 5]
# out2.shape [3, 3, 5]

out0, out1, out2 = fluid.layers.split(input, num_or_sections=[2, 3, 4], dim=1)
# out0.shape [3, 2, 5]
# out1.shape [3, 3, 5]
# out2.shape [3, 4, 5]

out0, out1, out2 = fluid.layers.split(input, num_or_sections=[2, 3, -1], dim=1)
# out0.shape [3, 2, 5]
# out1.shape [3, 3, 5]
# out2.shape [3, 4, 5]

# dim is negative, the real dim is (rank(input) + axis) which real
# value is 1.
out0, out1, out2 = fluid.layers.split(input, num_or_sections=3, dim=-2)
# out0.shape [3, 3, 5]
# out1.shape [3, 3, 5]
# out2.shape [3, 3, 5]