import paddle

type(paddle.to_tensor(1))
# <class 'paddle.Tensor'>

paddle.to_tensor(1)
# Tensor(shape=[1], dtype=int64, place=CPUPlace, stop_gradient=True,
#        [1])

x = paddle.to_tensor(1, stop_gradient=False)
print(x)
# Tensor(shape=[1], dtype=int64, place=CPUPlace, stop_gradient=False,
#        [1])

paddle.to_tensor(x)  # A new tensor will be created with default stop_gradient=True
# Tensor(shape=[1], dtype=int64, place=CPUPlace, stop_gradient=True,
#        [1])

paddle.to_tensor([[0.1, 0.2], [0.3, 0.4]], place=paddle.CPUPlace(), stop_gradient=False)
# Tensor(shape=[2, 2], dtype=float32, place=CPUPlace, stop_gradient=False,
#        [[0.10000000, 0.20000000],
#         [0.30000001, 0.40000001]])

type(paddle.to_tensor([[1+1j, 2], [3+2j, 4]], dtype='complex64'))
# <class 'paddle.Tensor'>

paddle.to_tensor([[1+1j, 2], [3+2j, 4]], dtype='complex64')
# Tensor(shape=[2, 2], dtype=complex64, place=CPUPlace, stop_gradient=True,
#        [[(1+1j), (2+0j)],
#         [(3+2j), (4+0j)]])