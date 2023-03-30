import paddle

data=paddle.to_tensor([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
#Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
#       [[(1+1j), (2+2j), (3+3j)],
#        [(4+4j), (5+5j), (6+6j)]])

conj_data=paddle.conj(data)
#Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
#       [[(1-1j), (2-2j), (3-3j)],
#        [(4-4j), (5-5j), (6-6j)]])