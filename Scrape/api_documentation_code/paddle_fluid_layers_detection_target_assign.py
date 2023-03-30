import paddle.fluid as fluid
import paddle
paddle.enable_static()
x = fluid.data(
    name='x',
    shape=[4, 20, 4],
    dtype='float',
    lod_level=1)
matched_id = fluid.data(
    name='indices',
    shape=[8, 20],
    dtype='int32')
trg, trg_weight = fluid.layers.target_assign(
    x,
    matched_id,
    mismatch_value=0)