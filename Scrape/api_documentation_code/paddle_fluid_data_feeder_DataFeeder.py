import paddle
paddle.enable_static()
import numpy as np
import paddle
import paddle.fluid as fluid

place = fluid.CPUPlace()
def reader():
    for _ in range(4):
        yield np.random.random([4]).astype('float32'), np.random.random([3]).astype('float32'),

main_program = fluid.Program()
startup_program = fluid.Program()

with fluid.program_guard(main_program, startup_program):
    data_1 = fluid.data(name='data_1', shape=[None, 2, 2], dtype='float32')
    data_2 = fluid.data(name='data_2', shape=[None, 1, 3], dtype='float32')
    out = fluid.layers.fc(input=[data_1, data_2], size=2)
    # ...
feeder = fluid.DataFeeder([data_1, data_2], place)

exe = fluid.Executor(place)
exe.run(startup_program)

feed_data = feeder.feed(reader())

# print feed_data to view feed results
# print(feed_data['data_1'])
# print(feed_data['data_2'])

outs = exe.run(program=main_program,
                feed=feed_data,
                fetch_list=[out])
print(outs)