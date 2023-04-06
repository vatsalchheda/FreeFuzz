import paddle

x = paddle.to_tensor([[[5,8,9,5],
                       [0,0,1,7],
                       [6,9,2,4]],
                      [[5,2,4,2],
                       [4,7,7,9],
                       [1,7,0,6]]],
                    dtype='float32')
out1 = paddle.argsort(x, axis=-1)
out2 = paddle.argsort(x, axis=0)
out3 = paddle.argsort(x, axis=1)

print(out1)
#[[[0 3 1 2]
#  [0 1 2 3]
#  [2 3 0 1]]
# [[1 3 2 0]
#  [0 1 2 3]
#  [2 0 3 1]]]

print(out2)
#[[[0 1 1 1]
#  [0 0 0 0]
#  [1 1 1 0]]
# [[1 0 0 0]
#  [1 1 1 1]
#  [0 0 0 1]]]

print(out3)
#[[[1 1 1 2]
#  [0 0 2 0]
#  [2 2 0 1]]
# [[2 0 2 0]
#  [1 1 0 2]
#  [0 2 1 1]]]