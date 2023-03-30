import paddle

x = paddle.to_tensor([[1, 2], [7, 8]])
y = paddle.to_tensor([[5, 6], [3, 4]])
res = paddle.subtract(x, y)
print(res)
# Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[-4, -4],
#         [ 4,  4]])

x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
y = paddle.to_tensor([1, 0, 4])
res = paddle.subtract(x, y)
print(res)
# Tensor(shape=[1, 2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[[ 0,  2, -1],
#          [ 0,  2, -1]]])

x = paddle.to_tensor([2, float('nan'), 5], dtype='float32')
y = paddle.to_tensor([1, 4, float('nan')], dtype='float32')
res = paddle.subtract(x, y)
print(res)
# Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [1. , nan, nan])

x = paddle.to_tensor([5, float('inf'), -float('inf')], dtype='float64')
y = paddle.to_tensor([1, 4, 5], dtype='float64')
res = paddle.subtract(x, y)
print(res)
# Tensor(shape=[3], dtype=float64, place=Place(cpu), stop_gradient=True,
#        [ 4.  ,  inf., -inf.])