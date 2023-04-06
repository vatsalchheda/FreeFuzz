import paddle.fluid as fluid
import numpy

with fluid.dygraph.guard():
    nodes_vector = numpy.random.random((1, 10, 5)).astype('float32')
    edge_set = numpy.random.random((1, 9, 2)).astype('int32')
    treeConv = fluid.dygraph.nn.TreeConv(
      feature_size=5, output_size=6, num_filters=1, max_depth=2)
    ret = treeConv(fluid.dygraph.base.to_variable(nodes_vector), fluid.dygraph.base.to_variable(edge_set))