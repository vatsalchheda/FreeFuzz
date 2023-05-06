import paddle
paddle.enable_static()

import paddle.fluid as fluid 
import numpy as np

x = fluid.data(name="x", shape=[None, 1], dtype="int32", lod_level=1) 
tree_info = [[0,0,0,1,2],[0,1,0,3,4],[0,1,0,5,6], [0,2,1,0,0],[1,2,1,0,0],[2,2,2,0,0],[3,2,2,0,0]]
tree_info_np = np.array(tree_info) 
tree_info_np = np.reshape(tree_info_np, (7,5)) 
node_nums = 7 
child_nums = 2 
child, leaf_mask = fluid.contrib.layers.tdm_child(x, node_nums, child_nums, param_attr=fluid.ParamAttr(initializer=fluid.initializer.NumpyArrayInitializer(tree_info_np)))
place = fluid.CPUPlace() 
exe = fluid.Executor(place) 
exe.run(fluid.default_startup_program()) 
xx = np.array([[2],[3]]).reshape((2,1)).astype("int32") 
child_res, leaf_mask_res = exe.run(feed={"x":xx}, fetch_list=[child, leaf_mask])