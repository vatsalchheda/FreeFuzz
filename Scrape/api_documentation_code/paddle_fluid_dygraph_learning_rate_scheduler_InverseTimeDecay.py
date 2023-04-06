import paddle.fluid as fluid
base_lr = 0.1
with fluid.dygraph.guard():
    emb = fluid.dygraph.Embedding([10, 10])
    sgd_optimizer = fluid.optimizer.SGD(
        learning_rate=fluid.dygraph.InverseTimeDecay(
              learning_rate=base_lr,
              decay_steps=10000,
              decay_rate=0.5,
              staircase=True),
        parameter_list = emb.parameters())