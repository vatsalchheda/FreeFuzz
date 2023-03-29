import paddle

DATATYPE = "float32"

anchor = paddle.rand(shape=(18, 6), dtype=DATATYPE)
positive = paddle.rand(shape=(18, 6), dtype=DATATYPE)
labels = paddle.rand(shape=(18,), dtype=DATATYPE)

npair_loss = paddle.nn.functional.npair_loss(anchor, positive, labels, l2_reg = 0.002)
print(npair_loss)