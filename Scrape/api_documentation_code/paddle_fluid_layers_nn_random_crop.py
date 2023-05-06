import paddle
paddle.enable_static()
import paddle.fluid as fluid
img = fluid.data("img", [None, 3, 256, 256])
# cropped_img is [-1, 3, 224, 224]
cropped_img = fluid.layers.random_crop(img, shape=[3, 224, 224])

# cropped_img2 shape: [-1, 2, 224, 224]
# cropped_img2 = fluid.layers.random_crop(img, shape=[2, 224, 224])

# cropped_img3 shape: [-1, 3, 128, 224]
# cropped_img3 = fluid.layers.random_crop(img, shape=[128, 224])