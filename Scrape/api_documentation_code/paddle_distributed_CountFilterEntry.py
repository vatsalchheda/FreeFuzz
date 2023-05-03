import paddle
paddle.enable_static()
import paddle

paddle.enable_static()

sparse_feature_dim = 1024
embedding_size = 64

entry = paddle.distributed.CountFilterEntry(10)

input = paddle.static.data(name='ins', shape=[1], dtype='int64')

emb = paddle.static.nn.sparse_embedding(
    input=input,
    size=[sparse_feature_dim, embedding_size],
    is_test=False,
    entry=entry,
    param_attr=paddle.ParamAttr(name="SparseFeatFactors",
                               initializer=paddle.nn.initializer.Uniform()))
