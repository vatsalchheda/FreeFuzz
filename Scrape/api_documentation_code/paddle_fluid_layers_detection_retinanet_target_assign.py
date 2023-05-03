import paddle
paddle.enable_static()
import paddle.fluid as fluid
bbox_pred = fluid.data(name='bbox_pred', shape=[1, 100, 4],
                  dtype='float32')
cls_logits = fluid.data(name='cls_logits', shape=[1, 100, 10],
                  dtype='float32')
anchor_box = fluid.data(name='anchor_box', shape=[100, 4],
                  dtype='float32')
anchor_var = fluid.data(name='anchor_var', shape=[100, 4],
                  dtype='float32')
gt_boxes = fluid.data(name='gt_boxes', shape=[10, 4],
                  dtype='float32')
gt_labels = fluid.data(name='gt_labels', shape=[10, 1],
                  dtype='int32')
is_crowd = fluid.data(name='is_crowd', shape=[1],
                  dtype='int32')
im_info = fluid.data(name='im_info', shape=[1, 3],
                  dtype='float32')
score_pred, loc_pred, score_target, loc_target, bbox_inside_weight, fg_num = fluid.layers.retinanet_target_assign(bbox_pred, cls_logits, anchor_box,
      anchor_var, gt_boxes, gt_labels, is_crowd, im_info, 10)