import paddle

x = paddle.to_tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])

# compute conditional number when p is None
out = paddle.linalg.cond(x)
# Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [1.41421342])

# compute conditional number when order of the norm is 'fro'
out_fro = paddle.linalg.cond(x, p='fro')
# Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [3.16227770])

# compute conditional number when order of the norm is 'nuc'
out_nuc = paddle.linalg.cond(x, p='nuc')
# Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [9.24263859])

# compute conditional number when order of the norm is 1
out_1 = paddle.linalg.cond(x, p=1)
# Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [2.])

# compute conditional number when order of the norm is -1
out_minus_1 = paddle.linalg.cond(x, p=-1)
# Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [1.])

# compute conditional number when order of the norm is 2
out_2 = paddle.linalg.cond(x, p=2)
# Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [1.41421342])

# compute conditional number when order of the norm is -1
out_minus_2 = paddle.linalg.cond(x, p=-2)
# Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [0.70710683])

# compute conditional number when order of the norm is inf
out_inf = paddle.linalg.cond(x, p=float("inf"))
# Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [2.])

# compute conditional number when order of the norm is -inf
out_minus_inf = paddle.linalg.cond(x, p=-float("inf"))
# Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [1.])

a = paddle.randn([2, 4, 4])
# Tensor(shape=[2, 4, 4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[-0.06784091, -0.07095790,  1.31792855, -0.58959651],
#          [ 0.20818676, -0.85640615, -0.89998871, -1.47439921],
#          [-0.49132481,  0.42250812, -0.77383220, -2.19794774],
#          [-0.33551720, -1.70003879, -1.09795380, -0.63737559]],

#         [[ 1.12026262, -0.16119350, -1.21157813,  2.74383283],
#          [-0.15999718,  0.18798758, -0.69392562,  1.35720372],
#          [-0.53013402, -2.26304483,  1.40843511, -1.02288902],
#          [ 0.69533503,  2.05261683, -0.02251151, -1.43127477]]])

a_cond_fro = paddle.linalg.cond(a, p='fro')
# Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [8.86691189 , 75.23817444])

b = paddle.randn([2, 3, 4])
# Tensor(shape=[2, 3, 4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[-0.43754861,  1.80796063, -0.78729683, -1.82264030],
#          [-0.27670753,  0.06620564,  0.29072434, -0.31155765],
#          [ 0.34123746, -0.05444612,  0.05001324, -1.46877074]],

#         [[-0.64331555, -1.51103854, -1.26277697, -0.68024760],
#          [ 2.59375715, -1.06665540,  0.96575671, -0.73330832],
#          [-0.47064447, -0.23945692, -0.95150250, -1.07125998]]])
b_cond_2 = paddle.linalg.cond(b, p=2)
# Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [6.64228773, 3.89068866])