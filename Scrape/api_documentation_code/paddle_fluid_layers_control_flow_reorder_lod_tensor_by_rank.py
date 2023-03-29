import paddle.fluid as fluid
data_desc = (['input', [9], 0], ['ref', [5], 1])
data = fluid.layers.data(name=data_desc[0][0], shape=data_desc[0][1])
rank_data = fluid.layers.data(name=data_desc[1][0], shape=data_desc[1][1])
table = fluid.layers.control_flow.lod_rank_table(rank_data)
new_data = fluid.layers.reorder_lod_tensor_by_rank(
                 x=data, rank_table=table)