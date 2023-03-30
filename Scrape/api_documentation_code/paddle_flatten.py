import paddle

image_shape=(2, 3, 4, 4)

x = paddle.arange(end=image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3])
img = paddle.reshape(x, image_shape)

out = paddle.flatten(img, start_axis=1, stop_axis=2)
# out shape is [2, 12, 4]

# out shares data with img in dygraph mode
img[0, 0, 0, 0] = -1
print(out[0, 0, 0]) # [-1]