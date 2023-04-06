import paddle

# example 1:
# dtype is None and the dtype of x is float16
x = paddle.zeros((1,2)).astype("float16")
out1 = paddle.randint_like(x, low=-5, high=5)
print(out1)
print(out1.dtype)
# [[0, -3]]  # random
# paddle.float16

# example 2:
# dtype is None and the dtype of x is float32
x = paddle.zeros((1,2)).astype("float32")
out2 = paddle.randint_like(x, low=-5, high=5)
print(out2)
print(out2.dtype)
# [[0, -3]]  # random
# paddle.float32

# example 3:
# dtype is None and the dtype of x is float64
x = paddle.zeros((1,2)).astype("float64")
out3 = paddle.randint_like(x, low=-5, high=5)
print(out3)
print(out3.dtype)
# [[0, -3]]  # random
# paddle.float64

# example 4:
# dtype is None and the dtype of x is int32
x = paddle.zeros((1,2)).astype("int32")
out4 = paddle.randint_like(x, low=-5, high=5)
print(out4)
print(out4.dtype)
# [[0, -3]]  # random
# paddle.int32

# example 5:
# dtype is None and the dtype of x is int64
x = paddle.zeros((1,2)).astype("int64")
out5 = paddle.randint_like(x, low=-5, high=5)
print(out5)
print(out5.dtype)
# [[0, -3]]  # random
# paddle.int64

# example 6:
# dtype is float64 and the dtype of x is float32
x = paddle.zeros((1,2)).astype("float32")
out6 = paddle.randint_like(x, low=-5, high=5, dtype="float64")
print(out6)
print(out6.dtype)
# [[0, -1]]  # random
# paddle.float64

# example 7:
# dtype is bool and the dtype of x is float32
x = paddle.zeros((1,2)).astype("float32")
out7 = paddle.randint_like(x, low=-5, high=5, dtype="bool")
print(out7)
print(out7.dtype)
# [[0, -1]]  # random
# paddle.bool

# example 8:
# dtype is int32 and the dtype of x is float32
x = paddle.zeros((1,2)).astype("float32")
out8 = paddle.randint_like(x, low=-5, high=5, dtype="int32")
print(out8)
print(out8.dtype)
# [[0, -1]]  # random
# paddle.int32

# example 9:
# dtype is int64 and the dtype of x is float32
x = paddle.zeros((1,2)).astype("float32")
out9 = paddle.randint_like(x, low=-5, high=5, dtype="int64")
print(out9)
print(out9.dtype)
# [[0, -1]]  # random
# paddle.int64

# example 10:
# dtype is int64 and the dtype of x is bool
x = paddle.zeros((1,2)).astype("bool")
out10 = paddle.randint_like(x, low=-5, high=5, dtype="int64")
print(out10)
print(out10.dtype)
# [[0, -1]]  # random
# paddle.int64