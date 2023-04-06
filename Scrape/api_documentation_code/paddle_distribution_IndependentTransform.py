import paddle

x = paddle.to_tensor([[1., 2., 3.], [4., 5., 6.]])

# Exponential transform with event_rank = 1
multi_exp = paddle.distribution.IndependentTransform(
    paddle.distribution.ExpTransform(), 1)
print(multi_exp.forward(x))
# Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[2.71828175  , 7.38905621  , 20.08553696 ],
#         [54.59814835 , 148.41316223, 403.42880249]])
print(multi_exp.forward_log_det_jacobian(x))
# Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [6. , 15.])