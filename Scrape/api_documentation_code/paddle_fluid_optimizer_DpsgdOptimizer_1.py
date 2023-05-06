import paddle
paddle.enable_static()
import paddle.fluid as fluid
import numpy

# First create the Executor.
place = fluid.CPUPlace() # fluid.CUDAPlace(0)
exe = fluid.Executor(place)

train_program = fluid.Program()
startup_program = fluid.Program()
with fluid.program_guard(train_program, startup_program):
    data = fluid.layers.data(name='X', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = fluid.layers.mean(hidden)
    optimizer = fluid.optimizer.Dpsgd(learning_rate=0.01, clip=10.0, batch_size=16.0, sigma=1.0)
    optimizer.minimize(loss)

# Run the startup program once and only once.
exe.run(startup_program)

x = numpy.random.random(size=(10, 1)).astype('float32')
outs = exe.run(program=train_program,
              feed={'X': x},
               fetch_list=[loss.name])
