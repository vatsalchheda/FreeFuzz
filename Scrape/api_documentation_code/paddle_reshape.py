import paddle

x = paddle.rand([2, 4, 6], dtype="float32")
positive_four = paddle.full([1], 4, "int32")

out = paddle.reshape(x, [-1, 0, 3, 2])
print(out)
# the shape is [2,4,3,2].

out = paddle.reshape(x, shape=[positive_four, 12])
print(out)
# the shape of out_2 is [4, 12].

shape_tensor = paddle.to_tensor([8, 6], dtype=paddle.int32)
out = paddle.reshape(x, shape=shape_tensor)
print(out.shape)
# the shape is [8, 6].
# out shares data with x in dygraph mode
x[0, 0, 0] = 10.
print(out[0, 0])
# the value is [10.]