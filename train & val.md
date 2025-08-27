https://github.com/zcablii/LSKNet/issues/90
现在主流的setting的把dota数据集的trainval放在一起作为训练集训12 epochs

先对图片进行切片 tools\data\dota\split\img_split.py

使用
bash tools/dist_train.sh configs/lsknet/lsk_s_fpn_1x_dota_le90.py 3
进行训练

使用
python tools/train.py configs/lsknet/lsk_s_fpn_1x_dota_le90.py --auto-resume
断点续训

使用
python tools/test.py configs/lsknet/lsk_s_fpn_1x_dota_le90.py work_dirs/lsk_s_fpn_1x_dota_le90/latest.pth --eval mAP
在trainval集上进行评估