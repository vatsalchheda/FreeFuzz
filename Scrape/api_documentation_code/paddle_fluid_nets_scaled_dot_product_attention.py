import paddle.fluid as fluid
import paddle
paddle.enable_static()

queries = fluid.data(name="queries", shape=[3, 5, 9], dtype="float32")
keys = fluid.data(name="keys", shape=[3, 6, 9], dtype="float32")
values = fluid.data(name="values", shape=[3, 6, 10], dtype="float32")
contexts = fluid.nets.scaled_dot_product_attention(queries, keys, values)
contexts.shape  # [3, 5, 10]