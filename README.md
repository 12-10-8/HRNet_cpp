# HRNet_cpp
HRNet的C++实现，编译器是VS2019，具体可以看博客[HRNet C++版本](https://jiahui.blog.csdn.net/article/details/116138711)

关于环境的配置可以自行百度。主要是opencv的cuda版和onnx推理库的配置。该项目是Release x64版本

exe文件、dll文件以及转换的onnx模型文件打包上传到了百度网盘中，链接：[HRNet-cpp_Release.zip](https://pan.baidu.com/s/1etultBMihiBx1vMAxxYHpw)  提取码：8fkr

使用方法：

```
HRNet-C++.exe -v E:\ytxs111.mp4 -m 0
```
参数的含义如下，-h可以查看

```
Usage: example [options...]
Options:
    -v, --video            Video path
    -c, --camera           camera index
    -m, --model            model type,0-w48_256x192,1-w48_384x288,2-w32_256x192,3-w32_128x96
    -d, --display          point display mode, 0-左右,1-左,2-右
    -w, --write_video      write video path
    -h, --help             Shows this page
```
