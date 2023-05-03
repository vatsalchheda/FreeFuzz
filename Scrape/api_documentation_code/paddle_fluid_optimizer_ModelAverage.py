import paddle
paddle.enable_static()
import paddle.fluid as fluid
import numpy

# First create the Executor.
place = fluid.CPUPlace()  # fluid.CUDAPlace(0)
exe = fluid.Executor(place)

train_program = fluid.Program()
startup_program = fluid.Program()
with fluid.program_guard(train_program, startup_program):
    # build net
    data = fluid.data(name='X', shape=[None, 1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = fluid.layers.mean(hidden)
    optimizer = fluid.optimizer.Momentum(learning_rate=0.2, momentum=0.1)
    optimizer.minimize(loss)

    # build ModelAverage optimizer
    model_average = fluid.optimizer.ModelAverage(0.15,
                                                 min_average_window=10000,
                                                 max_average_window=12500)

    exe.run(startup_program)
    for i in range(12500):
        x = numpy.random.random(size=(10, 1)).astype('float32')
        outs = exe.run(program=train_program,
                       feed={'X': x},
                       fetch_list=[loss.name])

    # apply ModelAverage
    with model_average.apply(exe):
        x = numpy.random.random(size=(10, 1)).astype('float32')
        exe.run(program=train_program,
                feed={'X': x},
                fetch_list=[loss.name])