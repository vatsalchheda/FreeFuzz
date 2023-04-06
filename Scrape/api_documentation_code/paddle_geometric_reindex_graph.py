import paddle

x = [0, 1, 2]
neighbors = [8, 9, 0, 4, 7, 6, 7]
count = [2, 3, 2]
x = paddle.to_tensor(x, dtype="int64")
neighbors = paddle.to_tensor(neighbors, dtype="int64")
count = paddle.to_tensor(count, dtype="int32")
reindex_src, reindex_dst, out_nodes = paddle.geometric.reindex_graph(x, neighbors, count)
# reindex_src: [3, 4, 0, 5, 6, 7, 6]
# reindex_dst: [0, 0, 1, 1, 1, 2, 2]
# out_nodes: [0, 1, 2, 8, 9, 4, 7, 6]