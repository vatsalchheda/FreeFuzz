import paddle
paddle.disable_static()

cell = paddle.nn.SimpleRNNCell(16, 32)

inputs = paddle.rand((4, 23, 16))
prev_h = paddle.randn((4, 32))
outputs, final_states = paddle.fluid.layers.rnn(cell, inputs, prev_h)