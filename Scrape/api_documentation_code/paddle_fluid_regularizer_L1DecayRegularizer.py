import paddle
paddle.enable_static()
# Example1: set Regularizer in optimizer
import paddle.fluid as fluid

main_prog = fluid.Program()
startup_prog = fluid.Program()
with fluid.program_guard(main_prog, startup_prog):
    data = fluid.layers.data(name='image', shape=[3, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    hidden = fluid.layers.fc(input=data, size=128, act='relu')
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
optimizer = fluid.optimizer.Adagrad(
    learning_rate=1e-4,
    regularization=fluid.regularizer.L1DecayRegularizer(
        regularization_coeff=0.1))
optimizer.minimize(avg_loss)


# Example2: set Regularizer both in ParamAttr and optimizer
import paddle.fluid as fluid

l1 = fluid.regularizer.L1Decay(regularization_coeff=0.1)
l2 = fluid.regularizer.L2Decay(regularization_coeff=0.1)
x = fluid.layers.uniform_random([3,4])

# set L1 regularization in fluid.ParamAttr
w_param = fluid.ParamAttr(regularizer=l1)
hidden1 = fluid.layers.fc(x, 8, param_attr=w_param)  # fc_0.w_0(L1), fc_0.b_0
hidden2 = fluid.layers.fc(hidden1, 16, param_attr=w_param)  # fc_1.w_0(L1), fc_1.b_0
predict = fluid.layers.fc(hidden2, 32)   # fc_3.w_0, fc_3.b_0
avg_loss = fluid.layers.mean(predict)

# set L2 regularization in optimizer
optimizer = fluid.optimizer.SGD(learning_rate=1e-4, regularization=l2)
optimizer.minimize(avg_loss)

# it will Print Message:
# Regularization of [fc_0.w_0, fc_1.w_0] have been set by ParamAttr or WeightNormParamAttr already.
# So, the Regularization of Optimizer will not take effect for these parameters!