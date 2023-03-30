import paddle

rnn = paddle.nn.LSTM(16, 32, 2)

x = paddle.randn((4, 23, 16))
prev_h = paddle.randn((2, 4, 32))
prev_c = paddle.randn((2, 4, 32))
y, (h, c) = rnn(x, (prev_h, prev_c))

print(y.shape)
print(h.shape)
print(c.shape)

#[4,23,32]
#[2,4,32]
#[2,4,32]