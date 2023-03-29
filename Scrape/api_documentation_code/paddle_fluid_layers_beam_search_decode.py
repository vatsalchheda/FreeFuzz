import paddle.fluid as fluid
import paddle
paddle.enable_static()
# Suppose `ids` and `scores` are LodTensorArray variables reserving
# the selected ids and scores of all steps
ids = fluid.layers.create_array(dtype='int64')
scores = fluid.layers.create_array(dtype='float32')
finished_ids, finished_scores = fluid.layers.beam_search_decode(
    ids, scores, beam_size=5, end_id=0)