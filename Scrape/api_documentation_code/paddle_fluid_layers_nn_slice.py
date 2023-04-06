import paddle

input = paddle.rand(shape=[4, 5, 6], dtype='float32')
# example 1:
# attr starts is a list which doesn't contain tensor.
axes = [0, 1, 2]
starts = [-3, 0, 2]
ends = [3, 2, 4]
sliced_1 = paddle.slice(input, axes=axes, starts=starts, ends=ends)
# sliced_1 is input[0:3, 0:2, 2:4].

# example 2:
# attr starts is a list which contain tensor.
minus_3 = paddle.full([1], -3, "int32")
sliced_2 = paddle.slice(input, axes=axes, starts=[minus_3, 0, 2], ends=ends)
# sliced_2 is input[0:3, 0:2, 2:4].