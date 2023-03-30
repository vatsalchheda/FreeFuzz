import paddle.fluid as fluid
import paddle

paddle.enable_static()

pb = fluid.data(name='prior_box', shape=[10, 4], dtype='float32')
pbv = fluid.data(name='prior_box_var', shape=[10, 4], dtype='float32')
loc = fluid.data(name='target_box', shape=[2, 21, 4], dtype='float32')
scores = fluid.data(name='scores', shape=[2, 21, 10], dtype='float32')
nmsed_outs, index = fluid.layers.detection_output(scores=scores,
                           loc=loc,
                           prior_box=pb,
                           prior_box_var=pbv,
                           return_index=True)