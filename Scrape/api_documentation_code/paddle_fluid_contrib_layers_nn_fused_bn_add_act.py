import paddle
paddle.enable_static()
import paddle
import paddle.fluid as fluid

def build_program(main_program, startup_program):
  with fluid.program_guard(main_program, startup_program):
    x = fluid.layers.data(name='x', shape=[1, 28, 28], dtype='float32') 
    y = fluid.layers.data(name="y", shape=[1], dtype='int64') 
    conv1_1 = fluid.layers.conv2d(input=x, filter_size=3, num_filters=32, stride=1, 
                                  padding=1, act=None, bias_attr=False, data_format='NHWC')
    conv1_2 = fluid.layers.conv2d(input=x, filter_size=3, num_filters=32, stride=1, padding=1, act=None, bias_attr=False, data_format='NHWC')
    bn = fluid.layers.batch_norm(input=conv1_1, act=None, data_layout='NHWC')
    fused_bn_add_act = fluid.contrib.layers.fused_bn_add_act(conv1_2, bn) 
    prediction = fluid.layers.fc(input=fused_bn_add_act, size=10, act='softmax') 
    loss = fluid.layers.cross_entropy(input=prediction, label=y) 
    loss = fluid.layers.mean(loss) 
    sgd = fluid.optimizer.SGD(learning_rate=0.001) 
    sgd = fluid.contrib.mixed_precision.decorate(sgd, use_dynamic_loss_scaling=True, init_loss_scaling=128.0)
    sgd.minimize(loss)
    return x, y, loss
  
iters = 5 
batch_size = 16 
support_gpu = fluid.is_compiled_with_cuda() 
if support_gpu:
  main_program = fluid.Program() 
  startup_program = fluid.Program() 
  place = fluid.CUDAPlace(0) 
  x, y, loss = build_program(main_program, startup_program)
  feeder = fluid.DataFeeder(feed_list=[x, y], place=place) 
  train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=batch_size)
  exe = fluid.Executor(place) 
  scope = fluid.Scope() 
  with fluid.scope_guard(scope):
    exe.run(startup_program) 
    for _ in range(iters):
      data = next(train_reader()) 
      loss_v = exe.run(main_program, feed=feeder.feed(data), fetch_list=[loss])