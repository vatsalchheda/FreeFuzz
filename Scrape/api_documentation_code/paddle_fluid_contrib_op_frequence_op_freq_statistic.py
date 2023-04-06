>>> import paddle.fluid as fluid
>>> uni_op_freq, adj_2_op_freq = fluid.contrib.op_freq_statistic(
>>>        fluid.default_main_program())
>>> for op_type, op_num in uni_op_freq:
>>>     print("%s         %d" % (op_type, op_num))
>>> for op_type, op_num in adj_2_op_freq:
>>>     print("%s         %d" % (op_type, op_num))