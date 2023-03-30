import paddle.fluid as fluid
boxes = fluid.data(name='bboxes', shape=[None, 81, 8],
                          dtype='float32')
scores = fluid.data(name='scores', shape=[None, 1, 81],
                          dtype='float32')
out = fluid.layers.locality_aware_nms(bboxes=boxes,
                                  scores=scores,
                                  score_threshold=0.5,
                                  nms_top_k=400,
                                  nms_threshold=0.3,
                                  keep_top_k=200,
                                  normalized=False)