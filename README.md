

# 基于云平台的智慧农田提取


### 训练步骤

#### 训练自己的数据集

1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的SegmentationClass中。    
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。    
4、在训练前利用voc_annotation.py文件生成对应的txt。    
5、在train.py文件夹下面，选择自己要使用的模型或添加自己的模型。   
6、注意修改train.py的num_classes为分类个数+1。    
7、运行train.py即可开始训练。  

### 预测步骤

先在inferenceNet.py中配置下参数，然后在inference.py里面可进行模型推理。    

### 生成onnx文件

produce_onnx.py 为导出并测试onnx文件



### 提取效果如下图：

![img](/imgs/img.jpg)