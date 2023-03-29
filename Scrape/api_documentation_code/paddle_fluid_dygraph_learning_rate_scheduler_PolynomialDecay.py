import paddle.fluid as fluid
start_lr = 0.01
total_step = 5000
end_lr = 0
with fluid.dygraph.guard():
    emb = fluid.dygraph.Embedding( [10, 10])
    optimizer  = fluid.optimizer.SGD(
        learning_rate = fluid.dygraph.PolynomialDecay(
        start_lr, total_step, end_lr, power=1.0),
        parameter_list = emb.parameters())