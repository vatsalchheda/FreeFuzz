import paddle

x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
indexes = paddle.to_tensor([[0, 1], [1, 2], [2, 1], [0, 0]], dtype="int32")
src_index, dst_index = indexes[:, 0], indexes[:, 1]
out = paddle.geometric.send_u_recv(x, src_index, dst_index, reduce_op="sum")
# Outputs: [[0., 2., 3.], [2., 8., 10.], [1., 4., 5.]]

x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
indexes = paddle.to_tensor([[0, 1], [2, 1], [0, 0]], dtype="int32")
src_index, dst_index = indexes[:, 0], indexes[:, 1]
out_size = paddle.max(dst_index) + 1
out = paddle.geometric.send_u_recv(x, src_index, dst_index, reduce_op="sum", out_size=out_size)
# Outputs: [[0., 2., 3.], [[2., 8., 10.]]]

x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
indexes = paddle.to_tensor([[0, 1], [2, 1], [0, 0]], dtype="int32")
src_index, dst_index = indexes[:, 0], indexes[:, 1]
out = paddle.geometric.send_u_recv(x, src_index, dst_index, reduce_op="sum")
# Outputs: [[0., 2., 3.], [2., 8., 10.], [0., 0., 0.]]