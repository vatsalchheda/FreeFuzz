import paddle
def reader():
    for i in range(10):
        yield i
batch_reader = paddle.batch(reader, batch_size=2)

for data in batch_reader():
    print(data)

# Output is
# [0, 1]
# [2, 3]
# [4, 5]
# [6, 7]
# [8, 9]