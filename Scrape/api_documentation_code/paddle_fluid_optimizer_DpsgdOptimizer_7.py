import paddle.fluid as fluid

with fluid.dygraph.guard():
    emb = fluid.dygraph.Embedding([10, 10])

    adam = fluid.optimizer.Adam(0.001, parameter_list=emb.parameters())
    state_dict = adam.state_dict()
