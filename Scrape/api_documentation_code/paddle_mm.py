import paddle
input = paddle.arange(1, 7).reshape((3, 2)).astype('float32')
mat2 = paddle.arange(1, 9).reshape((2, 4)).astype('float32')
out = paddle.mm(input, mat2)
print(out)
#        [[11., 14., 17., 20.],
#         [23., 30., 37., 44.],
#         [35., 46., 57., 68.]])