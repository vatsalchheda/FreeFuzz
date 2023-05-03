import paddle

iou_shape = [64, 32, 32]
num_classes = 5
predict = paddle.randint(low=0, high=255, shape=iou_shape, dtype='int64')
label = paddle.randint(low=0, high=255, shape=iou_shape, dtype='int64')
mean_iou, out_wrong, out_correct = paddle.fluid.layers.nn.mean_iou(predict, label, num_classes)