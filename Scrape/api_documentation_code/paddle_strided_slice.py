import paddle
x = paddle.zeros(shape=[3,4,5,6], dtype="float32")
# example 1:
# attr starts is a list which doesn't contain Tensor.
axes = [1, 2, 3]
starts = [-3, 0, 2]
ends = [3, 2, 4]
strides_1 = [1, 1, 1]
strides_2 = [1, 1, 2]
sliced_1 = paddle.strided_slice(x, axes=axes, starts=starts, ends=ends, strides=strides_1)
# sliced_1 is x[:, 1:3:1, 0:2:1, 2:4:1].
# example 2:
# attr starts is a list which contain tensor Tensor.
minus_3 = paddle.full(shape=[1], fill_value=-3, dtype='int32')
sliced_2 = paddle.strided_slice(x, axes=axes, starts=[minus_3, 0, 2], ends=ends, strides=strides_2)
# sliced_2 is x[:, 1:3:1, 0:2:1, 2:4:2].