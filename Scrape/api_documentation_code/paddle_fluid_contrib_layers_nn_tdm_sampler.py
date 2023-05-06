import paddle
paddle.enable_static()

import paddle.fluid as fluid 
import numpy as np 
x = fluid.data(name="x", shape=[None, 1], dtype="int32", lod_level=1) 
travel_list = [[1, 3], [1, 4], [2, 5], [2, 6]] # leaf node’s travel path, shape(leaf_node_num, layer_num) 
layer_list_flat = [[1], [2], [3], [4], [5], [6]] # shape(node_nums, 1)

neg_samples_num_list = [0, 0] # negative sample 
nums = 0 
layer_node_num_list = [2, 4] #two layer (exclude root node) 
leaf_node_num = 4

travel_array = np.array(travel_list) 
layer_array = np.array(layer_list_flat)

sample, label, mask = fluid.contrib.layers.tdm_sampler(x, neg_samples_num_list, layer_node_num_list, leaf_node_num, 
                                                       tree_travel_attr=fluid.ParamAttr(initializer=fluid.initializer.NumpyArrayInitializer(travel_array)),
                                                       tree_layer_attr=fluid.ParamAttr(initializer=fluid.initializer.NumpyArrayInitializer(layer_array)),
                                                       output_positive=True, output_list=True, seed=0, tree_dtype='int32')

place = fluid.CPUPlace() 
exe = fluid.Executor(place) 
exe.run(fluid.default_startup_program()) 
xx = np.array([[0],[1]]).reshape((2,1)).astype("int32")

exe.run(feed={"x":xx})