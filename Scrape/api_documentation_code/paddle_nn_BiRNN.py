import paddle

cell_fw = paddle.nn.LSTMCell(16, 32)
cell_bw = paddle.nn.LSTMCell(16, 32)
rnn = paddle.nn.BiRNN(cell_fw, cell_bw)

inputs = paddle.rand((2, 23, 16))
outputs, final_states = rnn(inputs)

print(outputs.shape)
print(final_states[0][0].shape,len(final_states),len(final_states[0]))

#[4,23,64]
#[2,32] 2 2