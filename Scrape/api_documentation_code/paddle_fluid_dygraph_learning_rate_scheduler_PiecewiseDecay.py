import paddle.fluid as fluid
boundaries = [10000, 20000]
values = [1.0, 0.5, 0.1]
with fluid.dygraph.guard():
    emb = fluid.dygraph.Embedding( [10, 10] )
    optimizer = fluid.optimizer.SGD(
       learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 0),
       parameter_list = emb.parameters() )