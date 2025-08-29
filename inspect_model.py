# filepath: /home/cjy/Large-Selective-Kernel-Network/inspect_model.py
from mmdet.apis import init_detector
import mmrotate  # noqa: F401

# --- 配置你的模型信息 ---
config_file = 'configs/lsknet/lsk_s_fpn_1x_dota_le90.py'
# 权重文件是可选的，对于查看模型结构而言不是必须的，但为了完整性我们加上
checkpoint_file = 'checkpoints/lsk_s_fpn_1x_dota_le90_20230116-99749191.pth'

# --- 构建模型 ---
# device='cpu' 也可以，因为我们只看结构，不进行计算
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# --- 打印模型结构 ---
print(model)

print("\n模型结构打印完成！")

# --- 计算并打印参数量 ---
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")