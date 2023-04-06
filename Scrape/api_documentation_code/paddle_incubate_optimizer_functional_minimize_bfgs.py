import paddle

def func(x):
    return paddle.dot(x, x)

x0 = paddle.to_tensor([1.3, 2.7])
results = paddle.incubate.optimizer.functional.minimize_bfgs(func, x0)
print("is_converge: ", results[0])
print("the minimum of func is: ", results[2])
# is_converge:  is_converge:  Tensor(shape=[1], dtype=bool, place=Place(gpu:0), stop_gradient=True,
#        [True])
# the minimum of func is:  Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [0., 0.])