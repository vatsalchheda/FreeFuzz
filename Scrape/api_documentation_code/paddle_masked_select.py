import paddle

x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0]])
mask = paddle.to_tensor([[True, False, False, False],
                         [True, True, False, False],
                         [True, False, False, False]])
out = paddle.masked_select(x, mask)
#[1.0 5.0 6.0 9.0]