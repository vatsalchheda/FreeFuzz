import paddle.fluid as fluid
base_lr = 0.1
lr = fluid.layers.cosine_decay(
learning_rate = base_lr, step_each_epoch=10000, epochs=120)