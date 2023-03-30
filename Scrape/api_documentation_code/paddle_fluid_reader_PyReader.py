import paddle
import paddle.fluid as fluid
import numpy as np

ITER_NUM = 5
BATCH_SIZE = 10

def reader_creator_random_image(height, width):
    def reader():
        for i in range(ITER_NUM):
            yield np.random.uniform(low=0, high=255, size=[height, width]), \
                np.random.random_integers(low=0, high=9, size=[1])
    return reader

place = fluid.CPUPlace()
with fluid.dygraph.guard(place):
    py_reader = fluid.io.PyReader(capacity=2, return_list=True)
    user_defined_reader = reader_creator_random_image(784, 784)
    py_reader.decorate_sample_list_generator(
        paddle.batch(user_defined_reader, batch_size=BATCH_SIZE),
        place)
    for image, label in py_reader():
        relu = fluid.layers.relu(image)