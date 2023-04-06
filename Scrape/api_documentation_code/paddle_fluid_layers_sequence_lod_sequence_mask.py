import paddle

lengths = paddle.to_tensor([10, 9, 8])
mask = paddle.nn.functional.sequence_mask(lengths)

print(mask.numpy())
# [[1 1 1 1 1 1 1 1 1 1]
#  [1 1 1 1 1 1 1 1 1 0]
#  [1 1 1 1 1 1 1 1 0 0]]