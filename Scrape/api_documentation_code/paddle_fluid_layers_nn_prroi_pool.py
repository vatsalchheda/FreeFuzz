import paddle
paddle.enable_static()
## prroi_pool without batch_roi_num
import paddle.fluid as fluid
x = fluid.data(name='x', shape=[None, 490, 28, 28], dtype='float32')
rois = fluid.data(name='rois', shape=[None, 4], lod_level=1, dtype='float32')
pool_out = fluid.layers.prroi_pool(x, rois, 1.0, 7, 7)

## prroi_pool with batch_roi_num
batchsize=4
x2 = fluid.data(name='x2', shape=[batchsize, 490, 28, 28], dtype='float32')
rois2 = fluid.data(name='rois2', shape=[batchsize, 4], dtype='float32')
batch_rois_num = fluid.data(name='rois_nums', shape=[batchsize], dtype='int64')
pool_out2 = fluid.layers.prroi_pool(x2, rois2, 1.0, 7, 7, batch_roi_nums=batch_rois_num)