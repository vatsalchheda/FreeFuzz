import paddle

input1 = paddle.to_tensor([[1.6, 1.2, -0.5], [3.2, 2.6, -5.8]], 'float32')
input2 = paddle.to_tensor([[0.5, 0.5, -1.8], [2.3, -1.4, 1.1]], 'float32')
label = paddle.to_tensor([1, -1], 'int64')

cosine_embedding_loss = paddle.nn.CosineEmbeddingLoss(margin=0.5, reduction='mean')
output = cosine_embedding_loss(input1, input2, label)
print(output) # [0.21155193]

cosine_embedding_loss = paddle.nn.CosineEmbeddingLoss(margin=0.5, reduction='sum')
output = cosine_embedding_loss(input1, input2, label)
print(output) # [0.42310387]

cosine_embedding_loss = paddle.nn.CosineEmbeddingLoss(margin=0.5, reduction='none')
output = cosine_embedding_loss(input1, input2, label)
print(output) # [0.42310387, 0.        ]