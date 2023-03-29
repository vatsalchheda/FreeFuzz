import paddle

iinfo_uint8 = paddle.iinfo(paddle.uint8)
print(iinfo_uint8)
# paddle.iinfo(min=0, max=255, bits=8, dtype=uint8)
print(iinfo_uint8.min) # 0
print(iinfo_uint8.max) # 255
print(iinfo_uint8.bits) # 8
print(iinfo_uint8.dtype) # uint8