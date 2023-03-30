import paddle
import paddle.fluid as fluid
paddle.enable_static()

data = paddle.static.data(name='x', shape=[-1, 1], dtype='int64', lod_level=1)
data2 = paddle.static.data(name='y', shape=[-1, 1], dtype='int64', lod_level=1)
inputs = [data, data2]
embs = fluid.layers.nn._pull_box_sparse(input=inputs, size=11, is_distributed=True, is_sparse=True)

label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64", lod_level=1)
ones = fluid.layers.fill_constant_batch_size_like(input=label, shape=[-1, 1], dtype="int64", value=1)
show_clk = paddle.cast(paddle.concat([ones, label], axis=1), dtype='float32')
show_clk.stop_gradient = True

cvms = fluid.contrib.layers.fused_seqpool_cvm(embs, 'sum', show_clk)