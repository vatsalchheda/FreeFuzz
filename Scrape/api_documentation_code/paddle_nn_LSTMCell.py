import paddle

x = paddle.randn((4, 16))
prev_h = paddle.randn((4, 32))
prev_c = paddle.randn((4, 32))

cell = paddle.nn.LSTMCell(16, 32)
y, (h, c) = cell(x, (prev_h, prev_c))

print(y.shape)
print(h.shape)
print(c.shape)

#[4,32]
#[4,32]
#[4,32]