import paddle.fluid as fluid
import paddle

paddle.enable_static()
input = fluid.data(
    name="input", shape=[3, 4, 5, 6], dtype='float32')

# example 1:
# attr starts is a list which doesn't contain tensor Variable.
axes = [0, 1, 2]
starts = [-3, 0, 2]
ends = [3, 2, 4]
strides_1 = [1, 1, 1]
strides_2 = [1, 1, 2]
sliced_1 = fluid.layers.strided_slice(input, axes=axes, starts=starts, ends=ends, strides=strides_1)
# sliced_1 is input[:, 0:3:1, 0:2:1, 2:4:1].


# example 2:
# attr starts is a list which contain tensor Variable.
minus_3 = fluid.layers.fill_constant([1], "int32", -3)
sliced_2 = fluid.layers.strided_slice(input, axes=axes, starts=[minus_3, 0, 2], ends=ends, strides=strides_2)
# sliced_2 is input[:, 0:3:1, 0:2:1, 2:4:2].