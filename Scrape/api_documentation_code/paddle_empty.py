import paddle

paddle.set_device("cpu")  # and use cpu device

# example 1: argument ``shape`` is a list which doesn't contain Tensor.
data1 = paddle.empty(shape=[2, 3], dtype='float32')
print(data1)
# Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[0.00000000, 0.        , 0.00000000],
#         [0.        , 0.29652897, 0.09356152]])       # uninitialized

# example 2: argument ``shape`` is a Tensor, the data type must be int64 or int32.
shape_data = paddle.to_tensor([2, 3]).astype('int32')
data2 = paddle.empty(shape=shape_data, dtype='float32')
print(data2)
# Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[-0.50543123, -0.09872390, -0.92634487],
#         [-0.51007903, -0.02454148,  1.29315734]])    # uninitialized

# example 3: argument ``shape`` is a list which contains Tensor.
dim2 = paddle.to_tensor([3]).astype('int32')
data3 = paddle.empty(shape=[2, dim2], dtype='float32')
print(data3)
# Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[ 0.00000000,  0.        , -0.92634487],
#         [-0.51007903, -0.02454148,  1.29315734]])    # uninitialized