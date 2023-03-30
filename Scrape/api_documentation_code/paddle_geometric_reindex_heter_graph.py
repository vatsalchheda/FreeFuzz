import paddle

x = [0, 1, 2]
neighbors_a = [8, 9, 0, 4, 7, 6, 7]
count_a = [2, 3, 2]
x = paddle.to_tensor(x, dtype="int64")
neighbors_a = paddle.to_tensor(neighbors_a, dtype="int64")
count_a = paddle.to_tensor(count_a, dtype="int32")
neighbors_b = [0, 2, 3, 5, 1]
count_b = [1, 3, 1]
neighbors_b = paddle.to_tensor(neighbors_b, dtype="int64")
count_b = paddle.to_tensor(count_b, dtype="int32")
neighbors = [neighbors_a, neighbors_b]
count = [count_a, count_b]
reindex_src, reindex_dst, out_nodes = paddle.geometric.reindex_heter_graph(x, neighbors, count)
# reindex_src: [3, 4, 0, 5, 6, 7, 6, 0, 2, 8, 9, 1]
# reindex_dst: [0, 0, 1, 1, 1, 2, 2, 0, 1, 1, 1, 2]
# out_nodes: [0, 1, 2, 8, 9, 4, 7, 6, 3, 5]