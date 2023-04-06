import paddle

data_1 = paddle.to_tensor([1, 4, 5, 7])
value_1, indices_1 = paddle.topk(data_1, k=1)
print(value_1) # [7]
print(indices_1) # [3]

data_2 = paddle.to_tensor([[1, 4, 5, 7], [2, 6, 2, 5]])
value_2, indices_2 = paddle.topk(data_2, k=1)
print(value_2) # [[7], [6]]
print(indices_2) # [[3], [1]]

value_3, indices_3 = paddle.topk(data_2, k=1, axis=-1)
print(value_3) # [[7], [6]]
print(indices_3) # [[3], [1]]

value_4, indices_4 = paddle.topk(data_2, k=1, axis=0)
print(value_4) # [[2, 6, 5, 7]]
print(indices_4) # [[1, 1, 0, 0]]