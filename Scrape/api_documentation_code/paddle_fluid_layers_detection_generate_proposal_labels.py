import paddle
import paddle.fluid as fluid
paddle.enable_static()
rpn_rois = fluid.data(name='rpn_rois', shape=[None, 4], dtype='float32')
gt_classes = fluid.data(name='gt_classes', shape=[None, 1], dtype='int32')
is_crowd = fluid.data(name='is_crowd', shape=[None, 1], dtype='int32')
gt_boxes = fluid.data(name='gt_boxes', shape=[None, 4], dtype='float32')
im_info = fluid.data(name='im_info', shape=[None, 3], dtype='float32')
rois, labels, bbox, inside_weights, outside_weights = fluid.layers.generate_proposal_labels(
               rpn_rois, gt_classes, is_crowd, gt_boxes, im_info,
               class_nums=10)