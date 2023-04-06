import paddle
import paddle.nn as nn

input = paddle.to_tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]], dtype=paddle.float32)
# label elements in {1., -1.}
label = paddle.to_tensor([[-1, 1, -1], [1, 1, 1], [1, -1, 1]], dtype=paddle.float32)

hinge_embedding_loss = nn.HingeEmbeddingLoss(margin=1.0, reduction='none')
loss = hinge_embedding_loss(input, label)
print(loss)
# Tensor([[0., -2., 0.],
#         [0., -1., 2.],
#         [1., 1., 1.]])

hinge_embedding_loss = nn.HingeEmbeddingLoss(margin=1.0, reduction='mean')
loss = hinge_embedding_loss(input, label)
print(loss)
# Tensor([0.22222222])