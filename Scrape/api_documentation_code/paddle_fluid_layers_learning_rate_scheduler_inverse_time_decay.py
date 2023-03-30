import paddle.fluid as fluid
import paddle
paddle.enable_static()
base_lr = 0.1
sgd_optimizer = fluid.optimizer.SGD(
    learning_rate=fluid.layers.inverse_time_decay(
          learning_rate=base_lr,
          decay_steps=10000,
          decay_rate=0.5,
          staircase=True))