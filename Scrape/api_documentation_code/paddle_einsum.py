import paddle
paddle.seed(102)
x = paddle.rand([4])
y = paddle.rand([5])

# sum
print(paddle.einsum('i->', x))
# Tensor(shape=[], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#   1.95791852)

# dot
print(paddle.einsum('i,i->', x, x))
# Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#   [1.45936954])

# outer
print(paddle.einsum("i,j->ij", x, y))
# Tensor(shape=[4, 5], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#   [[0.00079869, 0.00120950, 0.00136844, 0.00187187, 0.00192194],
#    [0.23455200, 0.35519385, 0.40186870, 0.54970956, 0.56441545],
#    [0.11773264, 0.17828843, 0.20171674, 0.27592498, 0.28330654],
#    [0.32897076, 0.49817693, 0.56364071, 0.77099484, 0.79162055]])

A = paddle.rand([2, 3, 2])
B = paddle.rand([2, 2, 3])

# transpose
print(paddle.einsum('ijk->kji', A))
#  Tensor(shape=[2, 3, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#   [[[0.95649719, 0.49684682],
#     [0.80071914, 0.46258664],
#     [0.49814570, 0.33383518]],
#
#    [[0.07637714, 0.29374704],
#     [0.51470858, 0.51907635],
#     [0.99066722, 0.55802226]]])

# batch matrix multiplication
print(paddle.einsum('ijk, ikl->ijl', A,B))
# Tensor(shape=[2, 3, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#   [[[0.32172769, 0.50617385, 0.41394392],
#     [0.51736701, 0.49921003, 0.38730967],
#     [0.69078457, 0.42282537, 0.30161136]],
#
#    [[0.32043904, 0.18164253, 0.27810261],
#     [0.50226176, 0.24512935, 0.39881429],
#     [0.51476848, 0.23367381, 0.39229113]]])

# Ellipsis transpose
print(paddle.einsum('...jk->...kj', A))
# Tensor(shape=[2, 2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#   [[[0.95649719, 0.80071914, 0.49814570],
#     [0.07637714, 0.51470858, 0.99066722]],
#
#    [[0.49684682, 0.46258664, 0.33383518],
#     [0.29374704, 0.51907635, 0.55802226]]])

# Ellipsis batch matrix multiplication
print(paddle.einsum('...jk, ...kl->...jl', A,B))
# Tensor(shape=[2, 3, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#   [[[0.32172769, 0.50617385, 0.41394392],
#     [0.51736701, 0.49921003, 0.38730967],
#     [0.69078457, 0.42282537, 0.30161136]],
#
#    [[0.32043904, 0.18164253, 0.27810261],
#     [0.50226176, 0.24512935, 0.39881429],
#     [0.51476848, 0.23367381, 0.39229113]]])