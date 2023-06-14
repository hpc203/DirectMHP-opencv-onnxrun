# DirectMHP-opencv-onnxrun
分别使用OpenCV、ONNXRuntime部署DirectMHP：用于全范围角度2D多人头部姿势估计，包含C++和Python两种版本的程序
使用opencv部署时，加载的是resources_nopost文件夹里的文件，
使用onnxruntime部署时，加载的是resources_withpost文件夹里的文件，
它们的区别是resources_withpost文件夹里的onnx文件里包含了nms。

一共有186个文件，占据磁盘空间大小17.1G，
onnx文件在百度云盘，链接：https://pan.baidu.com/s/1MoWGjtebc-yakgpD15WUeA 
提取码：i1g1
