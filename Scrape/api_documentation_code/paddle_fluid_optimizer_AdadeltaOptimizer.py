import paddle
paddle.enable_static()
import paddle.fluid as fluid

image = fluid.data(name='image', shape=[None, 28], dtype='float32')
fc = fluid.layers.fc(image, size=10)
cost = fluid.layers.reduce_mean(fc)
optimizer = fluid.optimizer.Adadelta(
    learning_rate=0.0003, epsilon=1.0e-6, rho=0.95)

# optimizer_ops is a list of optimizer operators to update parameters
# params_grads is a list of (param, param_grad), where param is each
# parameter and param_grad is the gradient variable of param.
optimizer_ops, params_grads = optimizer.minimize(cost)