import paddle
import paddle.nn.functional as F
input = paddle.to_tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]], dtype=paddle.float32)
# label elements in {1., -1.}
label = paddle.to_tensor([[-1, 1, -1], [1, 1, 1], [1, -1, 1]], dtype=paddle.float32)
loss = F.multi_label_soft_margin_loss(input, label, reduction='none')
print(loss)
# Tensor([3.49625897, 0.71111226, 0.43989015])
loss = F.multi_label_soft_margin_loss(input, label, reduction='mean')
print(loss)
# Tensor([1.54908717])