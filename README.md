# Mamba-with-transform
replace SSM with s-transform
针对图像处理 即插即用的MLLA Block
例如，在yolo中注册
'''
elif m is MLLABlock:
  c1, c2 = ch[f], args[0]
  if c2 != no:
    c2 = make_divisible(c2 * gw, 8)
  args = [c1, c2]
'''
![Image text](https://github.com/ZHjiuang/Mamba-with-transform/blob/main/mlla.png)
