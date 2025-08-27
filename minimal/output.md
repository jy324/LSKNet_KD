I will search for MidpointOffsetCoder to describe its encoding format. I will read the midpoint coder to describe its format. I will read DeltaXYWHAOBBoxCoder for 5-dim format. I will read the delta coder file to confirm format. 格式说明（区分两个阶段与训练/测试）：

RPN 阶段 (OrientedRPNHead):

1. 分类输出 (objectness)
   - 原始卷积输出形状（单尺度）: (N, A * 1, H, W) 其中 A=anchor数/位置 (这里3)
   - 若 use_sigmoid=True（当前配置）：对通道做 sigmoid 后得到每个 anchor 是否前景的概率 p_fg。
   - 拉平成一维后参与排序与 NMS 前筛选。
2. 回归输出
   - 形状: (N, A * 6, H, W)
   - 每个 anchor 6 个偏移 (dx, dy, dw, dh, da, db)
   - 编码/解码器：`MidpointOffsetCoder`
     - 输入编码：水平 anchor (x1,y1,x2,y2) 与 GT 旋转框 (cx,cy,w,h,angle)
     - 含义：
       dx,dy：GT 水平包围框中心相对 anchor 中心的归一化偏移
       dw,dh：log 尺度比
       da,db：利用 GT 多边形特定两个中点（以最高点 y_min 和最右点 x_max 方向推导的两个特征点 ga, gb）相对中心的归一化位置（见 coder 中 ga, gb 的求法）
   - 解码得到 proposals 时先转换为候选旋转框 (cx,cy,w,h,a, score)

RoI 阶段 (RotatedShared2FCBBoxHead):

1. 分类输出
   - 形状: (Total_RoIs, C+1) 其中 C=类别数 (这里15)，+1 是背景
   - 激活：训练时用 `CrossEntropyLoss`（内部 softmax），测试时取 softmax 后每类概率。
2. 回归输出
   - 形状（class-agnostic=True）: (Total_RoIs, 5)
   - 5 个偏移 (dx, dy, dw, dh, da)
   - 编码/解码器：`DeltaXYWHAOBBoxCoder`
     - Proposal / RoI 与 GT 旋转框 (cx,cy,w,h,a) 对应
     - dx,dy：中心坐标归一化偏移
     - dw,dh：log 尺度比
     - da：角度差（根据 angle_range=le90 规范化）
   - 解码后得到最终旋转框 (cx, cy, w, h, angle)

最终输出格式 (测试 simple_test):

- RPN proposals: Tensor shape (num_props, 6) = (cx, cy, w, h, angle, score) （score 是前景概率）
- 最终检测结果：
  - det_bboxes: (num_dets, 6) = (cx, cy, w, h, angle, score)
  - det_labels: (num_dets,) 每个为类别 id (0~C-1)
  - 若需要转四点多边形，可用转换函数（如 obb2poly）

训练时 target / pred 对应：

- RPN 回归目标：编码后 6 维，loss 对应 SmoothL1 于 (dx,dy,dw,dh,da,db)
- RoI 回归目标：编码后 5 维，loss 对应 SmoothL1 于 (dx,dy,dw,dh,da)

张量维度展开示例（单层 RPN）：

- cls: (N, 3, H, W) → permute+reshape → ((N*H*W*3), 1) 用 BCE
- reg: (N, 18, H, W) → permute+reshape → ((N*H*W*3), 6)

如还需要：

- 多类(class-aware)回归时形状差异
- 把旋转框转为 (x1,y1,x2,y2,x3,y3,x4,y4) 或水平框
  继续告诉我。
