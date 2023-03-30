import paddle

input = paddle.rand(shape=[10], dtype='float32')
label = paddle.rand(shape=[10], dtype='float32')
loss = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(input, label,
                                                ignore_index=-1, normalize=True)
print(loss)