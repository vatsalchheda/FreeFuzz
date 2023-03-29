import paddle
x1 = paddle.rand([1, 2, 3, 4]).astype('float32')
x2 = paddle.rand([1, 2, 1, 4]).astype('float32')
x3 = paddle.rand([1, 1, 3, 1]).astype('float32')
out1, out2, out3 = paddle.broadcast_tensors(input=[x1, x2, x3])
# out1, out2, out3: tensors broadcasted from x1, x2, x3 with shape [1,2,3,4]