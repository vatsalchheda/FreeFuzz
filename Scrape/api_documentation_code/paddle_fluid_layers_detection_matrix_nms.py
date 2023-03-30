import paddle.fluid as fluid
boxes = fluid.data(name='bboxes', shape=[None,81, 4],
                          dtype='float32', lod_level=1)
scores = fluid.data(name='scores', shape=[None,81],
                          dtype='float32', lod_level=1)
out = fluid.layers.matrix_nms(bboxes=boxes,
                              scores=scores,
                              background_label=0,
                              score_threshold=0.5,
                              post_threshold=0.1,
                              nms_top_k=400,
                              keep_top_k=200,
                              normalized=False)