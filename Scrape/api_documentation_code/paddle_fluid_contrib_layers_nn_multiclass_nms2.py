import paddle
paddle.enable_static()
import paddle.fluid as fluid
boxes = fluid.layers.data(name='bboxes', shape=[81, 4],
                          dtype='float32', lod_level=1)
scores = fluid.layers.data(name='scores', shape=[81],
                          dtype='float32', lod_level=1)
out, index = fluid.layers.multiclass_nms2(bboxes=boxes,
                                  scores=scores,
                                  background_label=0,
                                  score_threshold=0.5,
                                  nms_top_k=400,
                                  nms_threshold=0.3,
                                  keep_top_k=200,
                                  normalized=False,
                                  return_index=True)