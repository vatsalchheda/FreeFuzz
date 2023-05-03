import paddle
paddle.enable_static()
import paddle
import paddle.fluid as fluid
import paddle.dataset.mnist as mnist

def network(img, label):
    # User defined network. Here a simple regression as example
    predict = fluid.layers.fc(input=img, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=predict, label=label)
    return fluid.layers.mean(loss)

MEMORY_OPT = False
USE_CUDA = False

image = fluid.data(name='image', shape=[None, 1, 28, 28], dtype='float32')
label = fluid.data(name='label', shape=[None, 1], dtype='int64')
reader = fluid.layers.create_py_reader_by_data(capacity=64,
                                               feed_list=[image, label])
reader.decorate_paddle_reader(
    paddle.reader.shuffle(paddle.batch(mnist.train(), batch_size=5), buf_size=500))
img, label = fluid.layers.read_file(reader)
loss = network(img, label) # The definition of custom network and the loss function

place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

build_strategy = fluid.BuildStrategy()
build_strategy.memory_optimize = True if MEMORY_OPT else False
exec_strategy = fluid.ExecutionStrategy()
compiled_prog = fluid.compiler.CompiledProgram(
fluid.default_main_program()).with_data_parallel(
    loss_name=loss.name,
    build_strategy=build_strategy,
    exec_strategy=exec_strategy)

for epoch_id in range(2):
    reader.start()
try:
    while True:
        exe.run(compiled_prog, fetch_list=[loss.name])
except fluid.core.EOFException:
    reader.reset()