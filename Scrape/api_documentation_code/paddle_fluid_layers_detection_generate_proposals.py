import paddle.fluid as fluid
import paddle
paddle.enable_static()
scores = fluid.data(name='scores', shape=[None, 4, 5, 5], dtype='float32')
bbox_deltas = fluid.data(name='bbox_deltas', shape=[None, 16, 5, 5], dtype='float32')
im_info = fluid.data(name='im_info', shape=[None, 3], dtype='float32')
anchors = fluid.data(name='anchors', shape=[None, 5, 4, 4], dtype='float32')
variances = fluid.data(name='variances', shape=[None, 5, 10, 4], dtype='float32')
roi_probs = fluid.layers.generate_proposals(scores, bbox_deltas,
             im_info, anchors, variances)