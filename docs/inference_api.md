## LSKNet 推理 REST API 使用说明

该 API 基于 FastAPI & Uvicorn，封装 `tools/test.py` / `mmdet.apis.inference_detector` / `mmrotate.apis.inference_detector_by_patches`，支持单图推理与超大图滑窗推理。

### 设计原则与返回格式建议

推荐默认返回 **结构化 JSON (检测框 + 类别 + 置信度)**，并通过可选参数 `return_image=true` 再返回可视化后的图片（Base64 编码）。理由：

1. JSON 更轻量，便于二次处理（统计 / 过滤 / 入库）。
2. 前端可自行按需绘制（更灵活的样式 / 交互）。
3. Base64 图片体积较大；默认不返回可减少网络与延迟。

> 结论：默认仅返回检测结果（boxes + labels + scores），需要图片再显式指定。

### 启动服务

#### 1. 安装依赖

已在 `requirements/optional.txt` 添加：`fastapi`, `uvicorn[standard]`, `python-multipart`。

```bash
pip install -r requirements/optional.txt
```

（Windows PowerShell 同样适用。）

#### 2. 启动（Windows PowerShell 示例）

```powershell
python tools/inference_api.py `
  --config configs/lsknet/lsk_s_fpn_1x_dota_le90.py `
  --checkpoint ./data/pretrained/lsk_s_fpn_1x_dota_le90_20230116-99749191.pth `
  --device cuda:0 `
  --host 0.0.0.0 `
  --port 8000
```

#### 3. 启动（Linux / WSL / macOS 示例）

```bash
python tools/inference_api.py \
  --config configs/lsknet/lsk_s_fpn_1x_dota_le90.py \
  --checkpoint ./data/pretrained/lsk_s_fpn_1x_dota_le90_20230116-99749191.pth \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8000
```

启动后访问：

* Swagger 文档: `http://<host>:<port>/docs`
* OpenAPI JSON: `http://<host>:<port>/openapi.json`

### 接口说明

`POST /predict`  (Content-Type: multipart/form-data)

| 参数          | 类型           | 默认       | 说明                            |
| ------------- | -------------- | ---------- | ------------------------------- |
| image         | file           | 必填       | 输入图像文件                    |
| score_thr     | float          | 0.3        | 过滤阈值                        |
| patch_infer   | bool           | false      | 是否使用滑窗（超大图）          |
| return_image  | bool           | false      | 是否返回绘制后的 Base64 PNG     |
| sizes         | str(JSON list) | `[1024]` | 滑窗 patch 尺寸列表             |
| steps         | str(JSON list) | `[824]`  | 滑窗 stride 列表（与sizes对齐） |
| ratios        | str(JSON list) | `[1.0]`  | 多尺度缩放比例                  |
| merge_iou_thr | float          | 0.1        | 滑窗结果合并 IoU 阈值           |

#### 返回 JSON Schema

```json
{
  "time_ms": 42.7,
  "detections": [
    {"label": "plane", "score": 0.92, "bbox": [xc, yc, w, h, angle]},
    {"label": "ship",  "score": 0.88, "bbox": [xc, yc, w, h, angle]}
  ],
  "image_base64": "<可选>"
}
```

说明：

* 旋转框：`[xc, yc, w, h, angle]` （角度单位与训练配置一致，一般为弧度或度，取决于模型 head；若需可在此层转换）。
* 若为水平框模型则为 `[x1, y1, x2, y2]`。

### 调用示例

#### PowerShell (Windows)

```powershell
$resp = Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -Form @{
  image = Get-Item .\demo\dota_demo.jpg
  score_thr = '0.3'
  return_image = 'true'
}
$resp.time_ms
$resp.detections | Select-Object -First 3
```

保存可视化图片：

```powershell
[IO.File]::WriteAllBytes('out.png', [Convert]::FromBase64String($resp.image_base64))
```

#### curl (Linux / WSL)

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F image=@demo/dota_demo.jpg \
  -F score_thr=0.3 \
  -F return_image=true | jq '.'
```

#### Python 客户端

```python
import base64, requests

url = 'http://127.0.0.1:8000/predict'
files = { 'image': open('demo/dota_demo.jpg', 'rb') }
data = { 'score_thr': 0.35, 'return_image': 'true' }
r = requests.post(url, files=files, data=data, timeout=60)
resp = r.json()print('Latency(ms):', resp['time_ms'])
print('First det:', resp['detections'][:1])
if resp.get('image_base64'):
    with open('vis.png', 'wb') as f:
        f.write(base64.b64decode(resp['image_base64']))
```

### 超大图滑窗推理示例

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F image=@huge.png \
  -F patch_infer=true \
  -F sizes='[1024,1536]' \
  -F steps='[824,1200]' \
  -F ratios='[1.0,1.25]' \
  -F merge_iou_thr=0.15 \
  -F return_image=false
```

调参提示：

* `sizes` / `steps` 配对：同索引互相对应，多尺度生成所有组合。
* 步长略小于尺寸（例如 80%）可平衡覆盖与速度。
* `merge_iou_thr` 越低越容易合并，过低可能合并错误目标。

### 性能 & 优化

| 方向       | 手段                                                                                              |
| ---------- | ------------------------------------------------------------------------------------------------- |
| 启动预热   | 第一次请求延迟高，可在启动后自发一次空转推理                                                      |
| Batch 合并 | 目前接口单图；可扩展 `/predict_batch` 一次性多文件减少 IO                                       |
| GPU 利用   | 多进程 `--workers > 1` 前确认显存是否充足；模型已在进程内共享不可复用显存，需要单进程多请求优先 |
| 半精度     | 如果训练支持 FP16，可在初始化时修改 config 启用（当前脚本保持原样）                               |

### 常见错误

| 错误码 | 场景                           | 处理                            |
| ------ | ------------------------------ | ------------------------------- |
| 400    | 图像解析失败 / JSON 参数格式错 | 确认上传文件与 JSON 字符串格式  |
| 503    | 模型未加载                     | 确认服务是否以脚本方式正常启动  |
| 500    | patch 推理不可用               | 检查 mmrotate 版本 / 安装完整性 |

### Docker（可选示例）

在已有 `docker/Dockerfile` 基础上添加：

```dockerfile
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart
EXPOSE 8000
CMD ["python","tools/inference_api.py","--config","configs/lsknet/lsk_s_fpn_1x_dota_le90.py","--checkpoint","/workspace/./data/pretrained/lsk_s_fpn_1x_dota_le90_20230116-99749191.pth","--host","0.0.0.0","--port","8000"]
```

### 扩展建议

* `/predict_batch`：多文件 -> 返回列表或合并结果。
* JWT / Token 简易鉴权。
* 日志与请求耗时统计（middleware）。
* 指标上报（Prometheus / OpenTelemetry）。
* 模型热加载：增加 `/reload` 接口 + 原子替换全局模型。

### FAQ

**Q: 角度单位是度还是弧度？** 取决于当前模型 head 定义（如配置中的角度编码器），可在返回时统一转换。若需，可在 `_format_results` 中对角度字段乘/除 `np.pi/180`。

**Q: Base64 会很大怎么办？** 建议仅在调试或一次性展示时请求；生产前端可以拿 JSON 自绘。

**Q: 如何区分 HBB 与 OBB？** 通过每个框 ndarray 长度：6 -> 旋转 (含 score)；5 -> 水平 (含 score)。脚本已自动拆分。

---

若需要再增加批量接口或热加载支持，可继续提出需求。