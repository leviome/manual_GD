# manual_GD
手动实现梯度下降算法，训练MNIST数据集

---

Based on:<br>
1. Python 3.6
2. Numpy
3. Matplotlib
4. MNIST

---

通过到datasets路径下
```
python format_decoder.py
```

将MNIST数据集格式转换成`.npy`格式和`.txt`格式，这样方便手动操作。

然后，
```
python nn.py
```
就会开始训练。 
我在代码里面添加了指数衰减学习率等粗略优化方法，训练结果没有达到特别高的效果，0.9的准确率。后续将进一步进行优化。

