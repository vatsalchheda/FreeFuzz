import paddle.fluid as fluid
import paddle
paddle.enable_static()

# Suppose `probs` contains predicted results from the computation
# cell and `pre_ids` and `pre_scores` is the output of beam_search
# at previous step.
beam_size = 4
end_id = 1
pre_ids = fluid.data(
    name='pre_id', shape=[None, 1], lod_level=2, dtype='int64')
pre_scores = fluid.data(
    name='pre_scores', shape=[None, 1], lod_level=2, dtype='float32')
probs = fluid.data(
    name='probs', shape=[None, 10000], dtype='float32')
topk_scores, topk_indices = fluid.layers.topk(probs, k=beam_size)
accu_scores = fluid.layers.elementwise_add(
    x=fluid.layers.log(x=topk_scores),
    y=fluid.layers.reshape(pre_scores, shape=[-1]),
    axis=0)
selected_ids, selected_scores = fluid.layers.beam_search(
    pre_ids=pre_ids,
    pre_scores=pre_scores,
    ids=topk_indices,
    scores=accu_scores,
    beam_size=beam_size,
    end_id=end_id)