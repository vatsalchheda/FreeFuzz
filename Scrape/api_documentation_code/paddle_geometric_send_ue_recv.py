import paddle

x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
y = paddle.to_tensor([1, 1, 1, 1], dtype="float32")
indexes = paddle.to_tensor([[0, 1], [1, 2], [2, 1], [0, 0]], dtype="int32")
src_index, dst_index = indexes[:, 0], indexes[:, 1]
out = paddle.geometric.send_ue_recv(x, y, src_index, dst_index, message_op="add", reduce_op="sum")
# Outputs: [[1., 3., 4.], [4., 10., 12.], [2., 5., 6.]]

x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
y = paddle.to_tensor([1, 1, 1], dtype="float32")
indexes = paddle.to_tensor([[0, 1], [2, 1], [0, 0]], dtype="int32")
src_index, dst_index = indexes[:, 0], indexes[:, 1]
out_size = paddle.max(dst_index) + 1
out = paddle.geometric.send_ue_recv(x, y, src_index, dst_index, message_op="add", reduce_op="sum", out_size=out_size)
# Outputs: [[1., 3., 4.], [[4., 10., 12.]]]

x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
y = paddle.to_tensor([1, 1, 1], dtype="float32")
indexes = paddle.to_tensor([[0, 1], [2, 1], [0, 0]], dtype="int32")
src_index, dst_index = indexes[:, 0], indexes[:, 1]
out = paddle.geometric.send_ue_recv(x, y, src_index, dst_index, message_op="add", reduce_op="sum")
# Outputs: [[1., 3., 4.], [4., 10., 12.], [0., 0., 0.]]