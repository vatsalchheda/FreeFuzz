import paddle.fluid as fluid
base_lr = 0.1
with fluid.dygraph.guard():
    optimizer  = fluid.optimizer.SGD(
        learning_rate = fluid.dygraph.CosineDecay(
                base_lr, 10000, 120) )