import paddle.fluid as fluid
import paddle
paddle.enable_static()
boxes = fluid.data(name='bboxes', shape=[None,81, 4],
                          dtype='float32', lod_level=1)
scores = fluid.data(name='scores', shape=[None,81],
                          dtype='float32', lod_level=1)
out = fluid.layers.multiclass_nms(bboxes=boxes,
                                  scores=scores,
                                  background_label=0,
                                  score_threshold=0.5,
                                  nms_top_k=400,
                                  nms_threshold=0.3,
                                  keep_top_k=200,
                                  normalized=False)