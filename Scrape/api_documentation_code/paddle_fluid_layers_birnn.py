import paddle
paddle.disable_static()

cell_fw = paddle.nn.LSTMCell(16, 32)
cell_bw = paddle.nn.LSTMCell(16, 32)

inputs = paddle.rand((4, 23, 16))
hf, cf = paddle.rand((4, 32)), paddle.rand((4, 32))
hb, cb = paddle.rand((4, 32)), paddle.rand((4, 32))
initial_states = ((hf, cf), (hb, cb))
outputs, final_states = paddle.fluid.layers.birnn(
    cell_fw, cell_bw, inputs, initial_states)