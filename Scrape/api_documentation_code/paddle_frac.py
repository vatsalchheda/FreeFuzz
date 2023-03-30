import paddle

input = paddle.to_tensor([[12.22000003, -1.02999997],
                        [-0.54999995, 0.66000003]])
output = paddle.frac(input)
print(output)
# Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[ 0.22000003, -0.02999997],
#         [-0.54999995,  0.66000003]])