import paddle.fluid as fluid
import paddle
paddle.enable_static()
x = fluid.data(name='x', shape=[None, 255, 13, 13], dtype='float32')
img_size = fluid.data(name='img_size',shape=[None, 2],dtype='int64')
anchors = [10, 13, 16, 30, 33, 23]
boxes,scores = fluid.layers.yolo_box(x=x, img_size=img_size, class_num=80, anchors=anchors,
                                conf_thresh=0.01, downsample_ratio=32)