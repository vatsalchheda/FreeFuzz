import paddle.fluid as fluid
import paddle
paddle.enable_static()
multi_rois = []
multi_scores = []
for i in range(4):
    multi_rois.append(fluid.data(
        name='roi_'+str(i), shape=[None, 4], dtype='float32', lod_level=1))
for i in range(4):
    multi_scores.append(fluid.data(
        name='score_'+str(i), shape=[None, 1], dtype='float32', lod_level=1))

fpn_rois = fluid.layers.collect_fpn_proposals(
    multi_rois=multi_rois,
    multi_scores=multi_scores,
    min_level=2,
    max_level=5,
    post_nms_top_n=2000)