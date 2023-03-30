import paddle.fluid as fluid
warmup_steps = 100
learning_rate = 0.01
with fluid.dygraph.guard():
    emb = fluid.dygraph.Embedding([10, 10])
    optimizer  = fluid.optimizer.SGD(
        learning_rate = fluid.dygraph.NoamDecay(
               1/(warmup_steps *(learning_rate ** 2)),
               warmup_steps),
        parameter_list = emb.parameters())