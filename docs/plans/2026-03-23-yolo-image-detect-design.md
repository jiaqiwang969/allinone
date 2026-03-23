# YOLO 单图检测闭环设计

## 背景

当前 `allinone` 主链已经具备：

- `raw perception json -> standardized payload`
- `standardized payload -> runtime-observation`
- `runtime-observation -> guidance + Qwen 中文输出`

但 `raw perception json` 仍然依赖手工样例文件，`YOLO` 虽然已经有 adapter，却没有真正进入主链。

这会导致系统还停留在“人工伪造上游输入”的阶段，而不是“真实图片进入系统后自动得到指导结果”的阶段。

## 本轮目标

把第一段真实感知链打通：

`单张图片 -> YOLO -> raw perception json -> standardized payload -> runtime-observation`

本轮只做单张图片，不做视频，不做逐帧会话编排。

## 为什么先做单图

有三个选择：

1. 单图经过 `raw perception` 边界再进入主链
2. 单图直接跳过 `raw perception`，直接生成标准 payload
3. 直接上视频

本轮选 1，原因是：

- 它是最短可执行闭环
- 它保留了已经建立好的标准边界
- 后面接入 `V-JEPA` 时，不需要重写主链
- 后面扩展视频时，只需要在前面加“抽帧/采样/聚合”层

## 设计决策

### 1. 新增 `detect-image` CLI

新增命令：

`python3 -m allinone.interfaces.cli.main detect-image --image ... --model ... --targets ... --output ...`

职责：

- 读取单张图片
- 调用 `UltralyticsDetectorAdapter`
- 生成上游 raw perception JSON

### 2. 输出仍然是 raw perception JSON

输出结构保持与现有样例一致：

```json
{
  "detections": {
    "prediction_rows": [...],
    "image_size": [w, h],
    "target_labels": ["meter"]
  },
  "vjepa": {
    "visibility_score": 1.0,
    "readable_ratio": 1.0
  }
}
```

这样做的原因是：

- 不破坏已存在的 `build-observation-payload`
- 让 `YOLO` 和未来 `V-JEPA` 在上游阶段自然汇合
- 便于保存、回放、审计和研究比较

### 3. 第一版 `vjepa` 用占位默认值

本轮不做真实 `V-JEPA` 推理。

因此 `detect-image` 会先写出默认：

- `visibility_score = 1.0`
- `readable_ratio = 1.0`

这不是最终方案，但足够支持第一条真实图片闭环。

### 4. 图片尺寸自动读取

不要求用户手工传 `width/height`。

系统直接从图片文件读取尺寸，再传给 `YOLO adapter` 归一化和落盘。这样更接近真实使用方式，也减少人为输入错误。

## 分层边界

- `application`：新增“从单图检测结果构建 raw perception payload”的用例
- `infrastructure`：继续由 `UltralyticsDetectorAdapter` 负责真实检测
- `interfaces`：CLI 负责参数接收、文件读写和命令编排

本轮允许 `application` 继续沿用当前仓库已有模式，通过 adapter 参数注入完成编排，不在这一步重做更大的端口抽象。

## 数据流

1. 用户传入图片路径、模型路径、目标标签
2. 系统读取图片宽高
3. `YOLO` 运行检测，返回候选框
4. 用例把检测结果转换成 raw perception JSON
5. `build-observation-payload` 把 raw JSON 转成标准 payload
6. `runtime-observation` 产出 guidance 和语言提示

## 测试策略

按 TDD 做三层测试：

1. 用例测试
   验证检测结果能正确转成 raw perception JSON
2. CLI 测试
   验证 `detect-image` 能正确接参数并写出文件
3. 闭环 smoke 测试
   验证 `detect-image -> build-observation-payload -> runtime-observation` 可以串起来

真实 YOLO 模型推理本轮不在本地测试中强依赖，而是通过注入假 adapter 保证结构正确，再用服务器环境补真实验证。

## 非目标

本轮明确不做：

- 视频抽帧
- 逐帧时序聚合
- 真实 `V-JEPA` 质量评分
- 训练或微调 YOLO
- 会话级 evidence 自动沉淀

## 预期结果

完成后，`allinone` 就从“手工伪造输入的主链”升级成“真实工业图片可进入系统的主链”。

这一步的价值很直接：

- `YOLO` 真正接入 `allinone`
- `raw perception` 边界被验证有效
- 之后接 `V-JEPA` 和视频链路会更顺
