# VBGAN
在原有Bashlines的基础上修改了如下文件

在model中加入了ViT、DCGAN模型所需文件，并大量修改了vit.py、dcgan.py文件。

在model.data中，修改了CIB_data.py,dataloader.py文件

编写了VBGAN.py文件和VBGAN_ablation.py文件，其中VBGAN.py为改进后模型，VBGAN_ablation.py为报告中消融实验所用模型。

model文件夹为模型所引用的别的模型。

configs文件夹为配置信息

logs文件夹为日志文件夹

## How to run?
请下载ViT-B_16预训练参数并放置于checkpoint文件夹下，请下载cifar-10、NUS_WIDE数据集并放置于dataset文件夹下

根据自己的需求修改VBGAN.py置顶配置即可运行,无需修改其余文件