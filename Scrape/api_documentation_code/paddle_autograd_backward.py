import paddle
x = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32', stop_gradient=False)
y = paddle.to_tensor([[3, 2], [3, 4]], dtype='float32')

grad_tensor1 = paddle.to_tensor([[1,2], [2, 3]], dtype='float32')
grad_tensor2 = paddle.to_tensor([[1,1], [1, 1]], dtype='float32')

z1 = paddle.matmul(x, y)
z2 = paddle.matmul(x, y)

paddle.autograd.backward([z1, z2], [grad_tensor1, grad_tensor2], True)
print(x.grad)
#[[12. 18.]
# [17. 25.]]

x.clear_grad()

paddle.autograd.backward([z1, z2], [grad_tensor1, None], True)
print(x.grad)
#[[12. 18.]
# [17. 25.]]

x.clear_grad()

paddle.autograd.backward([z1, z2])
print(x.grad)
#[[10. 14.]
# [10. 14.]]