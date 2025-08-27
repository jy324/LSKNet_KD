import random
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List, Optional
import re
import pandas as pd
from copy import deepcopy
import torch.nn.utils.prune as prune
from torchinfo import summary
from mmrotate.models.backbones.lsknet import LSKNet
# lsknet.py 中需要 DWConv 类
from mmrotate.models.backbones.lsknet import DWConv
import argparse
import json
import os

def _prune_dwconv(dwconv_module: DWConv, keep_idx: torch.Tensor) -> DWConv:
    """ 通过重建一个通道数和分组数更少的卷积层来剪枝 DWConv 模块 """
    old_conv = dwconv_module.dwconv
    n_keep = keep_idx.numel()

    # 创建一个新的、更小的深度可分离卷积层
    new_conv = nn.Conv2d(
        in_channels=n_keep,
        out_channels=n_keep,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=n_keep,  # 关键：分组数必须等于输入/输出通道数
        bias=(old_conv.bias is not None),
    )

    # 复制保留下来的权重和偏置
    new_conv.weight.data = old_conv.weight.data[keep_idx].clone()
    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias.data[keep_idx].clone()

    # 将新的卷积层包装回 DWConv 模块
    new_dwconv_module = DWConv(dim=n_keep)
    new_dwconv_module.dwconv = new_conv
    return new_dwconv_module


# Network Analysis Tool
def analyze_network_layers(model: nn.Module, input_shape: Tuple[int, ...] = None,
                           device: str = 'cpu', mode: str = 'leaf_only') -> Dict[str, Any]:
    """
    分析PyTorch网络的层结构和参数分布，并自动标注需要避开的“残差敏感层”
    """
    model = model.to(device)
    model.eval()
    module_dict = dict(model.named_modules())

    # ---------- 识别残差敏感层（需剔除） ----------
    residual_sensitive = set()

    # 规则：1) 任意 downsample.*（shortcut分支的 conv/bn 等）直接标记为敏感
    for name, module in module_dict.items():
        if '.downsample.' in name and isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            residual_sensitive.add(name)

    # 2) ResNet Block 内主分支的“相加前的最后一层”标记为敏感：
    block_has_conv3 = set()
    block_has_conv2 = set()

    for name, module in module_dict.items():
        m = re.match(r'^(layer\d+\.\d+)\.(conv[1-3]|bn[1-3]|downsample\.\d+)$', name)
        if m:
            block_base, leaf = m.groups()
            if leaf == 'conv3':
                block_has_conv3.add(block_base)
            if leaf == 'conv2':
                block_has_conv2.add(block_base)

    for block_base in set(list(block_has_conv3) + list(block_has_conv2)):
        if f'{block_base}.conv3' in module_dict:
            for leaf in ['conv3', 'bn3']:
                name = f'{block_base}.{leaf}'
                if name in module_dict:
                    residual_sensitive.add(name)
        else:
            for leaf in ['conv2', 'bn2']:
                name = f'{block_base}.{leaf}'
                if name in module_dict:
                    residual_sensitive.add(name)

    # ---------- 统计层与参数 ----------
    layer_names = []
    layer_params = []
    layer_types = []
    total_params = sum(p.numel() for p in model.parameters())

    def is_leaf(m: nn.Module) -> bool:
        return len(list(m.children())) == 0

    if mode == 'leaf_only':
        for name, module in model.named_modules():
            if name == '':
                continue
            if is_leaf(module):
                layer_names.append(name)
                layer_types.append(type(module).__name__)
                layer_params.append(sum(p.numel() for p in module.parameters()))
    elif mode == 'direct_only':
        for name, module in model.named_modules():
            if name == '':
                continue
            layer_names.append(name)
            layer_types.append(type(module).__name__)
            layer_params.append(sum(p.numel() for p in module.parameters(recurse=False)))
    else:  # 'all'
        for name, module in model.named_modules():
            if name == '':
                continue
            layer_names.append(name)
            layer_types.append(type(module).__name__)
            layer_params.append(sum(p.numel() for p in module.parameters()))

    param_ratios = [params / total_params * 100 if total_params > 0 else 0 for params in layer_params]

    layer_info = []
    for i, (name, layer_type, params, ratio) in enumerate(zip(layer_names, layer_types, layer_params, param_ratios)):
        layer_info.append({
            'layer_index': i + 1,
            'layer_name': name,
            'layer_type': layer_type,
            'parameters': params,
            'param_ratio_percent': round(ratio, 4),
            'residual_sensitive': name in residual_sensitive  # 新增标记
        })

    layer_outputs = {}
    if input_shape is not None:
        try:
            layer_outputs = get_layer_output_shapes(model, input_shape, device)
        except Exception as e:
            print(f"无法获取层输出形状: {e}")

    return {
        'layer_names': layer_names,
        'layer_info': layer_info,
        'total_parameters': total_params,
        'layer_outputs': layer_outputs,
        'residual_sensitive': residual_sensitive,  # 新增：集合
        'summary_df': pd.DataFrame(layer_info)        # 含 residual_sensitive 列
    }

def get_layer_output_shapes(model: nn.Module, input_shape: Tuple[int, ...],
                            device: str = 'cpu') -> Dict[str, Tuple]:
    """
    获取网络中每层的输出形状
    """
    layer_outputs = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                layer_outputs[name] = tuple(output.shape)
            elif isinstance(output, (list, tuple)):
                layer_outputs[name] = tuple(out.shape if isinstance(out, torch.Tensor)
                                            else str(out) for out in output)
        return hook_fn

    # 注册钩子
    for name, module in model.named_modules():
        if name:  # 跳过根模块
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)

    # 前向传播
    with torch.no_grad():
        dummy_input = torch.randn(input_shape).to(device)
        model(dummy_input)

    # 移除钩子
    for hook in hooks:
        hook.remove()

    return layer_outputs

def print_layer_analysis(analysis_result: Dict[str, Any], top_n: int = 10):
    """
    打印层分析结果；默认从“参数最多的前 N 层”中剔除“残差敏感层”，
    返回可安全用于结构化通道剪枝的层名列表。
    """
    print("=" * 80)
    print("网络层分析报告")
    print("=" * 80)
    print(f"总参数量: {analysis_result['total_parameters']:,}")
    print(f"网络层数量: {len(analysis_result['layer_names'])}")

    df = analysis_result['summary_df'].copy()
    total_ratio = df['param_ratio_percent'].sum()
    print(f"参数占比总和: {total_ratio:.4f}%")

    # 过滤：剔除“残差敏感层”
    if 'residual_sensitive' in df.columns:
        before = len(df)
        df = df[~df['residual_sensitive']]
        removed = before - len(df)
        print(f"\n已剔除残差敏感层（可能导致维度不匹配的层）：{removed} 个")
    else:
        print("\n未检测到 residual_sensitive 字段，跳过过滤。")

    print()
    print(f"参数量最多且可安全剪枝的前{top_n}层:")
    print("-" * 80)

    # 只在可安全集合中选 top-n
    top_layers = df.sort_values('parameters', ascending=False).head(top_n)

    top_layerNames = []
    for _, row in top_layers.iterrows():
        print(f"{row['layer_name']:30} | {row['layer_type']:15} | "
              f"{row['parameters']:>10,} | {row['param_ratio_percent']:>8.4f}%")
        top_layerNames.append(row['layer_name'])

    print()
    print("完整层列表（含是否残差敏感标记）:")
    print("-" * 80)
    for i, row in analysis_result['summary_df'].iterrows():
        flag = " [RES]" if row.get('residual_sensitive', False) else ""
        print(f"{int(row['layer_index']):3d}. {row['layer_name']}{flag}")

    return top_layerNames

# Pruning function
def random_prune_layer(model: nn.Module, targetLayerNames: list, amount: float = 0.25, observe: bool = False):
    """
    随机剪枝指定层的参数
    """
    res_model = deepcopy(model)
    for name, module in res_model.named_modules():
        if name in targetLayerNames and isinstance(module, nn.Conv2d):
            org_params = deepcopy(module.weight.size())
            prune.random_unstructured(module, name='weight', amount=amount)  # 随机剪枝25%参数
            if observe:
                print(f"剪枝前 {name} 参数量: {org_params}, 剪枝后: {module.weight.size()}")
    print("随机剪枝完成！")
    return res_model

def structured_prune_layer(model: nn.Module, targetLayerNames: List[str], amount: float = 0.25, observe: bool = False,
                           seed: Optional[int] = None) -> nn.Module:
    """
    残差安全版结构化剪枝（仅在 ResNet block 内传播）。
    """
    def _get_module(root: nn.Module, dotted: str) -> nn.Module:
        mod = root
        for p in dotted.split('.'):
            mod = getattr(mod, p)
        return mod

    def _set_module(root: nn.Module, dotted: str, new_m: nn.Module) -> None:
        parent = root
        parts = dotted.split('.')
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_m)

    def _prune_conv_out(conv: nn.Conv2d, keep_idx: torch.Tensor) -> nn.Conv2d:
        if conv.groups != 1:
            raise ValueError(f"不支持分组卷积剪枝（groups={conv.groups}）")
        new_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=keep_idx.numel(),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
        )
        new_conv.weight.data = conv.weight.data[keep_idx].clone()
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data[keep_idx].clone()
        return new_conv

    def _prune_conv_in(conv: nn.Conv2d, keep_idx: torch.Tensor) -> nn.Conv2d:
        if conv.groups != 1:
            raise ValueError(f"不支持分组卷积剪枝（groups={conv.groups}）")
        new_conv = nn.Conv2d(
            in_channels=keep_idx.numel(),
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
        )
        new_conv.weight.data = conv.weight.data[:, keep_idx].clone()
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data.clone()
        return new_conv

    def _prune_bn(bn: nn.BatchNorm2d, keep_idx: torch.Tensor) -> nn.BatchNorm2d:
        new_bn = nn.BatchNorm2d(keep_idx.numel())
        new_bn.weight.data = bn.weight.data[keep_idx].clone()
        new_bn.bias.data = bn.bias.data[keep_idx].clone()
        new_bn.running_mean = bn.running_mean[keep_idx].clone()
        new_bn.running_var = bn.running_var[keep_idx].clone()
        return new_bn

    # 复制模型
    res_model = deepcopy(model)

    # 设置可复现性（按层扰动避免每层相同子集）
    base_gen = torch.Generator()
    if seed is not None:
        base_gen.manual_seed(seed)

    for target_name in targetLayerNames:
        module_dict = dict(res_model.named_modules())

        # 基础校验
        conv = module_dict.get(target_name, None)
        if not isinstance(conv, nn.Conv2d):
            if observe:
                print(f"⚠️ 跳过：{target_name} 不存在或不是 Conv2d")
            continue
        if conv.groups != 1:
            if observe:
                print(f"⚠️ 跳过分组卷积：{target_name} (groups={conv.groups})")
            continue

        out_ch = conv.out_channels
        n_keep = max(1, out_ch - int(round(out_ch * amount)))
        gen = base_gen
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed ^ (hash(target_name) & 0x7FFFFFFF))

        keep_idx = torch.randperm(out_ch, generator=gen)[:n_keep].sort().values

        # 判断是否为标准 ResNet 命名：layerX.Y.convZ
        m = re.match(r'^(layer\d+\.\d+)\.(conv\d+)$', target_name)
        if not m:
            # 非标准路径：专门处理 LSKNet 的 Mlp 模块或其他通用层
            if target_name.endswith('.mlp.fc1'):
                # 针对 LSKNet 中导致错误的特定模式进行处理
                try:
                    # 1. 剪枝 fc1 层的输出通道
                    new_conv = _prune_conv_out(conv, keep_idx)
                    _set_module(res_model, target_name, new_conv)
                    if observe:
                        print(f"✅ 剪枝 LSK Mlp 层: {target_name}  {out_ch}→{n_keep}")

                    # 2. 查找并级联剪枝后续的 dwconv 层
                    dwconv_name = target_name.replace('.fc1', '.dwconv')
                    dwconv_module = module_dict.get(dwconv_name)

                    if isinstance(dwconv_module, DWConv):
                        new_dwconv = _prune_dwconv(dwconv_module, keep_idx)
                        _set_module(res_model, dwconv_name, new_dwconv)
                        if observe:
                            print(f"  ↳ 级联更新: {dwconv_name}")
                    else:
                        if observe:
                            print(f"  ↳ ⚠️ 未找到匹配的 DWConv: {dwconv_name}")

                    # 3. 级联更新 fc2 层的输入通道
                    fc2_name = target_name.replace('.fc1', '.fc2')
                    fc2_module = module_dict.get(fc2_name)
                    if isinstance(fc2_module, nn.Conv2d):
                        new_fc2 = _prune_conv_in(fc2_module, keep_idx)
                        _set_module(res_model, fc2_name, new_fc2)
                        if observe:
                            print(f"  ↳ 级联更新: {fc2_name} in_channels")
                    else:
                        if observe:
                            print(f"  ↳ ⚠️ 未找到匹配的 fc2 层: {fc2_name}")
                except Exception as e:
                    print(f"❌ LSK Mlp 剪枝失败 {target_name}: {e}")

            continue  # 非标准路径处理结束

        # 标准 ResNet：解析 block 与叶子
        block_base, leaf = m.groups()  # e.g., block_base='layer1.0', leaf='conv1'
        is_bottleneck = (f"{block_base}.conv3" in module_dict)

        # 安全白/黑名单
        if 'downsample' in target_name:
            if observe:
                print(f"❌ 禁止剪残差接口相关层：{target_name}")
            continue
        if is_bottleneck:
            allowed, forbidden = {"conv1", "conv2"}, {"conv3"}
        else:
            allowed, forbidden = {"conv1"}, {"conv2"}

        if leaf in forbidden:
            if observe:
                print(f"❌ 禁止剪残差接口相关层：{target_name}")
            continue
        if leaf not in allowed:
            if observe:
                print(f"⚠️ 跳过不在安全白名单的层：{target_name}（允许：{allowed}）")
            continue

        # 1) 剪当前 conv 的输出 + 同级 bn（bn1/bn2）
        pruned_conv = _prune_conv_out(conv, keep_idx)
        _set_module(res_model, target_name, pruned_conv)

        bn_name = target_name.replace('conv', 'bn')
        if bn_name in module_dict and isinstance(module_dict[bn_name], nn.BatchNorm2d):
            bn = module_dict[bn_name]
            _set_module(res_model, bn_name, _prune_bn(bn, keep_idx))

        if observe:
            print(f"✅ 剪 {target_name}: {out_ch}→{n_keep}；同步 {(bn_name if bn_name in module_dict else '无BN')}")

        # 2) 仅在 block 内级联更新下一层卷积的“输入通道”
        if is_bottleneck:
            next_leaf = 'conv2' if leaf == 'conv1' else 'conv3'  # conv1→conv2；conv2→conv3
        else:
            next_leaf = 'conv2' if leaf == 'conv1' else None

        if next_leaf is not None:
            next_name = f"{block_base}.{next_leaf}"
            next_conv = module_dict.get(next_name, None)
            if isinstance(next_conv, nn.Conv2d) and next_conv.groups == 1:
                try:
                    _set_module(res_model, next_name, _prune_conv_in(next_conv, keep_idx))
                    if observe:
                        print(f"   ↳ 级联更新 {next_name}.in_channels = {n_keep}")
                except Exception as e:
                    print(f"❌ 更新 {next_name} 失败：{e}")
            elif observe:
                print(f"   ↳ 跳过级联：{next_name} 不存在或不是标准 Conv2d（或为分组卷积）")

    return res_model

def _parse_args():
    parser = argparse.ArgumentParser(description='LSKNet 预训练权重剪枝脚本 (main-prune)')
    parser.add_argument('--checkpoint', type=str, default='././data/pretrained/lsk_s_backbone.pth',
                        help='预训练 backbone 权重路径 (state_dict)')
    parser.add_argument('--img-size', type=int, default=1024, help='输入图像尺寸 (方形)')
    parser.add_argument('--amount', type=float, default=0.5, help='剪枝比例 (0~1) 针对选中层输出通道')
    parser.add_argument('--topk', type=int, default=10, help='按参数量排序选取前 K 个可安全层进行尝试剪枝')
    parser.add_argument('--mode', type=str, default='structured', choices=['structured', 'random'], help='剪枝方式')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (结构化随机保留通道)')
    parser.add_argument('--output', type=str, default='././data/pretrained/lsk_s_backbone_pruned.pth', help='输出剪枝后权重保存路径')
    parser.add_argument('--save-metadata', action='store_true', help='同时保存剪枝元数据 JSON')
    parser.add_argument('--no-summary', action='store_true', help='不打印模型结构 summary (可加速)')
    parser.add_argument('--dry-run', action='store_true', help='仅分析不实际剪枝/保存')
    return parser.parse_args()


def _build_lsknet(checkpoint: str, img_size: int) -> nn.Module:
    """构建并加载预训练 LSKNet backbone (仅 backbone state_dict)."""
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f'未找到 checkpoint: {checkpoint}')
    model = LSKNet(img_size=img_size, in_chans=3, embed_dims=[64, 128, 320, 512], drop_rate=0.0,
                   drop_path_rate=0.1, depths=[2, 2, 4, 2],
                   init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
                   norm_cfg=dict(type='SyncBN', requires_grad=True))
    return model


def _remove_pruning_reparam(model: nn.Module):
    """如果使用 torch.nn.utils.prune 引入了 mask 与 orig_weight, 需要恢复为普通参数."""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_orig') and hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
            if hasattr(module, 'bias_orig') and hasattr(module, 'bias_mask'):
                prune.remove(module, 'bias')


def _save(model: nn.Module, out_path: str, meta: Optional[Dict[str, Any]] = None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f'✅ 剪枝后权重已保存: {out_path}')
    if meta:
        meta_path = os.path.splitext(out_path)[0] + '_meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f'ℹ️  剪枝元数据保存: {meta_path}')


def _run_cli():
    args = _parse_args()
    print('==== LSKNet main-prune 启动 ====', flush=True)
    print(f'加载预训练: {args.checkpoint}')
    model = _build_lsknet(args.checkpoint, args.img_size)

    # 分析层
    analysis = analyze_network_layers(model, mode='leaf_only')
    top_layer_names = print_layer_analysis(analysis, top_n=args.topk)
    if args.dry_run:
        print('Dry-run 模式：结束 (未执行剪枝/保存)。')
        return

    original_params = analysis['total_parameters']

    if args.mode == 'structured':
        print(f'执行结构化剪枝: amount={args.amount}')
        pruned_model = structured_prune_layer(model, top_layer_names, amount=args.amount, observe=True, seed=args.seed)
    else:
        print(f'执行随机非结构化剪枝: amount={args.amount}')
        pruned_model = random_prune_layer(model, top_layer_names, amount=args.amount, observe=True)
        _remove_pruning_reparam(pruned_model)  # 清理 reparam 方便下游加载

    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    pruning_ratio = (1 - pruned_params / original_params) * 100

    print('\n' + '=' * 80)
    print('剪枝效果总结')
    print('-' * 80)
    print(f'原始模型总参数量: {original_params:,}')
    print(f'剪枝后模型总参数量: {pruned_params:,}')
    print(f'参数量压缩比例: {pruning_ratio:.2f}%')
    print('=' * 80 + '\n')

    if not args.no_summary:
        try:
            summary(pruned_model, (1, 3, args.img_size, args.img_size))
        except Exception as e:
            print(f'⚠️ summary 失败: {e}')

    meta = None
    if args.save_metadata:
        meta = dict(
            checkpoint=args.checkpoint,
            output=args.output,
            mode=args.mode,
            amount=args.amount,
            topk=args.topk,
            seed=args.seed,
            original_params=original_params,
            pruned_params=pruned_params,
            pruning_ratio_percent=pruning_ratio,
            pruned_target_layers=top_layer_names
        )

    _save(pruned_model, args.output, meta)
    print('完成。您可以在下游检测配置中将 init_cfg.checkpoint 指向新的剪枝权重。')


if __name__ == '__main__':
    _run_cli()

