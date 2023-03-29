import paddle.fluid as fluid

bboxes_low = fluid.data(
    name='bboxes_low', shape=[1, 44, 4], dtype='float32')
bboxes_high = fluid.data(
    name='bboxes_high', shape=[1, 11, 4], dtype='float32')
scores_low = fluid.data(
    name='scores_low', shape=[1, 44, 10], dtype='float32')
scores_high = fluid.data(
    name='scores_high', shape=[1, 11, 10], dtype='float32')
anchors_low = fluid.data(
    name='anchors_low', shape=[44, 4], dtype='float32')
anchors_high = fluid.data(
    name='anchors_high', shape=[11, 4], dtype='float32')
im_info = fluid.data(
    name="im_info", shape=[1, 3], dtype='float32')
nmsed_outs = fluid.layers.retinanet_detection_output(
    bboxes=[bboxes_low, bboxes_high],
    scores=[scores_low, scores_high],
    anchors=[anchors_low, anchors_high],
    im_info=im_info,
    score_threshold=0.05,
    nms_top_k=1000,
    keep_top_k=100,
    nms_threshold=0.45,
    nms_eta=1.0)