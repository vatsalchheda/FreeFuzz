import paddle
import paddle.nn as nn

x1 = paddle.to_tensor([[1., 2., 3.],
                    [2., 3., 4.]], dtype="float32")
x2 = paddle.to_tensor([[8., 3., 3.],
                    [2., 3., 4.]], dtype="float32")

cos_sim_func = nn.CosineSimilarity(axis=0)
result = cos_sim_func(x1, x2)
print(result)
# Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [0.65079135, 0.98058069, 1.        ])
