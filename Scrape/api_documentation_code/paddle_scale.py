# scale with parameter scale as a Tensor
import paddle

data = paddle.randn(shape=[2, 3], dtype='float32')
factor = paddle.to_tensor([2], dtype='float32')
res = paddle.scale(data, scale=factor, bias=1.0)