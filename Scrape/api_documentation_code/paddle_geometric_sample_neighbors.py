import paddle

# edges: (3, 0), (7, 0), (0, 1), (9, 1), (1, 2), (4, 3), (2, 4),
#        (9, 5), (3, 5), (9, 6), (1, 6), (9, 8), (7, 8)
row = [3, 7, 0, 9, 1, 4, 2, 9, 3, 9, 1, 9, 7]
colptr = [0, 2, 4, 5, 6, 7, 9, 11, 11, 13, 13]
nodes = [0, 8, 1, 2]
sample_size = 2
row = paddle.to_tensor(row, dtype="int64")
colptr = paddle.to_tensor(colptr, dtype="int64")
nodes = paddle.to_tensor(nodes, dtype="int64")
out_neighbors, out_count = paddle.geometric.sample_neighbors(row, colptr, nodes, sample_size=sample_size)