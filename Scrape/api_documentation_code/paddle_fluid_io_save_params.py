import paddle
import paddle.fluid as fluid


paddle.enable_static()
params_path = "./my_paddle_model"
image = fluid.data(name='img', shape=[None, 28, 28], dtype='float32')
label = fluid.data(name='label', shape=[None, 1], dtype='int64')
feeder = fluid.DataFeeder(feed_list=[image, label], place=fluid.CPUPlace())
predict = fluid.layers.fc(input=image, size=10, act='softmax')

loss = fluid.layers.cross_entropy(input=predict, label=label)
avg_loss = paddle.mean(loss)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
fluid.io.save_params(executor=exe, dirname=params_path)
# The parameters weights and bias of the fc layer in the network are going to
# be saved in different files in the path "./my_paddle_model"