import paddle

# In imperative mode:
# size x: (2, 2, 3) and y: (2, 3, 2)
x = paddle.to_tensor([[[1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0]],
                    [[3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0]]])
y = paddle.to_tensor([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],
                    [[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]])
out = paddle.bmm(x, y)
# Tensor(shape=[2, 2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[[6. , 6. ],
#          [12., 12.]],

#         [[45., 45.],
#          [60., 60.]]])