import paddle
import paddle.nn as nn

x0 = paddle.arange(3, 6).reshape((3, 1)).astype(paddle.int64)
w0 = paddle.full(shape=(10, 3), fill_value=2).astype(paddle.float32)

# x.data = [[3], [4], [5]]
# x.shape = [3, 1]
x = paddle.to_tensor(x0, stop_gradient=False)

# w.data = [[2. 2. 2.] ... [2. 2. 2.]]
# w.shape = [10, 3]
w = paddle.to_tensor(w0, stop_gradient=False)

# emb.data = [[[2., 2., 2.]], [[2., 2., 2.]], [[2., 2., 2.]]]
# emb.shape = [3, 1, 3]
emb = nn.functional.embedding(
        x=x, weight=w, sparse=True, name="embedding")