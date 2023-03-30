import paddle

ids = paddle.to_tensor([[[2, 2], [6, 1]], [[3, 9], [6, 1]], [[0, 1], [9, 0]]])

parents = paddle.to_tensor([[[0, 0], [1, 1]], [[1, 0], [1, 0]], [[0, 0], [0, 1]]])

final_sequences = paddle.nn.functional.gather_tree(ids, parents)
# [[[2, 2], [1, 6]], [[3, 3], [6, 1]], [[0, 1], [9, 0]]]