import paddle.fluid as fluid
data = fluid.data(name='data', shape=[None, 3, 32, 32])
feature_map_0 = fluid.layers.conv2d(data, num_filters=2,
                                    filter_size=3)
feature_map_1 = fluid.layers.conv2d(feature_map_0, num_filters=2,
                                    filter_size=1)
loss = fluid.layers.fsp_matrix(feature_map_0, feature_map_1)