import paddle
paddle.enable_static()
import paddle.fluid as fluid
pb = fluid.data(
               name='prior_box',
               shape=[10, 4],
               dtype='float32')
pbv = fluid.data(
               name='prior_box_var',
               shape=[10, 4],
               dtype='float32')
loc = fluid.data(name='target_box', shape=[10, 4], dtype='float32')
scores = fluid.data(name='scores', shape=[10, 21], dtype='float32')
gt_box = fluid.data(
     name='gt_box', shape=[4], lod_level=1, dtype='float32')
gt_label = fluid.data(
     name='gt_label', shape=[1], lod_level=1, dtype='float32')
loss = fluid.layers.ssd_loss(loc, scores, gt_box, gt_label, pb, pbv)