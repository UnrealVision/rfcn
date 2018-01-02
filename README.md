# R-FCN

当下同时保持速度快且精度高的目标检测算法，在这里我们开源这个框架，当然同样是基于caffe实现的，由于它需要一些层的支持，所以我们自己拉取了caffe版本，**unreal_caffe**, 大家可以在本组织的项目中找到它，按装之后你可能还需要训练它。

我们目前还有开源训练的脚本，因为官方的实现很冗长，这里大家可以参照其他的开源实现，本项目的目标是用C++实现一个推理加速，从而实现高精度高速度的检测框架。



## 使用

直接在根目录：

```
python demo.py -i images/
```

就会自动检测images下面的图片，但是直线你得获取到一个`rfcn`的caffemodel，放在output文件夹下。





## Copyright

如果你在项目中使用到了这个，不妨带上UnrealVision这个标识吧。