>>> import paddle.fluid as fluid
>>> x = fluid.data(name='x', shape=[None, 4], dtype='float32')
>>> y = fluid.data(name='y', shape=[None, 4], dtype='float32')
>>> iou = fluid.layers.iou_similarity(x=x, y=y)
>>> matched_indices, matched_dist = fluid.layers.bipartite_match(iou)