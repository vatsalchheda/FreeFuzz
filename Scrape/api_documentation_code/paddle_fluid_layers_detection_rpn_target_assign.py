import paddle
paddle.enable_static()
import paddle.fluid as fluid
bbox_pred = fluid.data(name='bbox_pred', shape=[None, 4], dtype='float32')
cls_logits = fluid.data(name='cls_logits', shape=[None, 1], dtype='float32')
anchor_box = fluid.data(name='anchor_box', shape=[None, 4], dtype='float32')
anchor_var = fluid.data(name='anchor_var', shape=[None, 4], dtype='float32')
gt_boxes = fluid.data(name='gt_boxes', shape=[None, 4], dtype='float32')
is_crowd = fluid.data(name='is_crowd', shape=[None], dtype='float32')
im_info = fluid.data(name='im_infoss', shape=[None, 3], dtype='float32')
loc, score, loc_target, score_target, inside_weight = fluid.layers.rpn_target_assign(
    bbox_pred, cls_logits, anchor_box, anchor_var, gt_boxes, is_crowd, im_info)