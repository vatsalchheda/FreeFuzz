import paddle.fluid as fluid
import paddle
paddle.enable_static()
x = fluid.data(name='x', shape=[None, 255, 13, 13], dtype='float32')
gt_box = fluid.data(name='gt_box', shape=[None, 6, 4], dtype='float32')
gt_label = fluid.data(name='gt_label', shape=[None, 6], dtype='int32')
gt_score = fluid.data(name='gt_score', shape=[None, 6], dtype='float32')
anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
anchor_mask = [0, 1, 2]
loss = fluid.layers.yolov3_loss(x=x, gt_box=gt_box, gt_label=gt_label,
                                gt_score=gt_score, anchors=anchors,
                                anchor_mask=anchor_mask, class_num=80,
                                ignore_thresh=0.7, downsample_ratio=32)