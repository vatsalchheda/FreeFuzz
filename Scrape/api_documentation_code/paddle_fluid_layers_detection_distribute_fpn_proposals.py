import paddle.fluid as fluid
import paddle
paddle.enable_static()
fpn_rois = fluid.data(
    name='data', shape=[None, 4], dtype='float32', lod_level=1)
multi_rois, restore_ind = fluid.layers.distribute_fpn_proposals(
    fpn_rois=fpn_rois,
    min_level=2,
    max_level=5,
    refer_level=4,
    refer_scale=224)