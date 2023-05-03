import paddle
paddle.enable_static()
import paddle.fluid as fluid

# 10 for max_node_size of dataset, 5 for vector width
nodes_vector = fluid.layers.data(
    name='vectors', shape=[10, 5], dtype='float32')
# 10 for max_node_size of dataset, 2 for every edge has two nodes
# edges must be directional
edge_set = fluid.layers.data(name='edge_set', shape=[
                             10, 2], dtype='float32')
# the shape of output will be [10, 6, 1],
# 10 for max_node_size of dataset, 6 for output size, 1 for 1 filter
out_vector = fluid.layers.tree_conv(nodes_vector, edge_set, 6, 1, 2)
out_vector = fluid.layers.reshape(out_vector, shape=[-1, 10, 6]) 
out_vector_2 = fluid.layers.tree_conv(out_vector, edge_set, 3, 4, 2)
pooled = fluid.layers.reduce_max(out_vector, dim=2) # global pooling