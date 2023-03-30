import paddle
import paddle.fluid as fluid

input = paddle.to_tensor([[2.0, 2.0], [2.0, 2.0]], dtype='float32')
reward = fluid.layers.clip_by_norm(x=input, max_norm=1.0)
# [[0.5, 0.5], [0.5, 0.5]]