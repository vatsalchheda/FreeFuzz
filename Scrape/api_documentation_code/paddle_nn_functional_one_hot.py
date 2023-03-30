import paddle
# Correspond to the first example above, where label.shape is 4 and one_hot_label.shape is [4, 4].
label = paddle.to_tensor([1, 1, 3, 0], dtype='int64')
# label.shape = [4]
one_hot_label = paddle.nn.functional.one_hot(label, num_classes=4)
# one_hot_label.shape = [4, 4]
# one_hot_label = [[0., 1., 0., 0.],
#                  [0., 1., 0., 0.],
#                  [0., 0., 0., 1.],
#                  [1., 0., 0., 0.]]