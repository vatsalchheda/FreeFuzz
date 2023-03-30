import paddle.fluid as fluid

with fluid.dygraph.guard():
    emb = fluid.dygraph.Embedding([10, 10])

    state_dict = emb.state_dict()
    fluid.save_dygraph( state_dict, "paddle_dy")

    adam = fluid.optimizer.Adam( learning_rate = fluid.layers.noam_decay( 100, 10000),
                                 parameter_list = emb.parameters() )

    state_dict = adam.state_dict()
    fluid.save_dygraph( state_dict, "paddle_dy")