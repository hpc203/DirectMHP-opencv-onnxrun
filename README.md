# DirectMHP-opencv-onnxrun
分别使用OpenCV、ONNXRuntime部署DirectMHP：用于全范围角度2D多人头部姿势估计，包含C++和Python两种版本的程序。

在使用opencv部署时，加载的是resources_nopost文件夹里的文件，注意在使用opencv-dnn推理时，opencv版本需要是4.7，
程序才能正常运行。注意opencv-dnn在读取
["directmhp_300wlp_m_finetune_HxW.onnx", "directmhp_300wlp_s_finetune_HxW.onnx", "directmhp_agora_m_HxW.onnx", "directmhp_agora_s_HxW.onnx", "directmhp_cmu_m_HxW.onnx", "directmhp_cmu_s_HxW.onnx"]
这些文件，会出错的。

在使用onnxruntime部署时，加载的是resources_withpost文件夹里的文件，
它们的区别是resources_withpost文件夹里的onnx文件里包含了nms。

一共有186个文件，占据磁盘空间大小17.1G，
onnx文件在百度云盘，链接：https://pan.baidu.com/s/1MoWGjtebc-yakgpD15WUeA 
提取码：i1g1

训练源码在https://github.com/hnuzhy/DirectMHP
