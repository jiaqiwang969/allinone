# Source Assets

## 当前确定要吸收的资产

### 1. autoresearch

- 位置：`/Users/jqwang/29-autoresearch/autoresearch`
- 价值：自动实验、结果对比、参数迭代、研究闭环
- 进入位置：`infrastructure/research/autoresearch`

### 2. ultralytics

- 位置：`/Users/jqwang/29-autoresearch/ultralytics`
- 价值：YOLO 检测、分割、跟踪能力
- 进入位置：`infrastructure/perception/yolo`

### 3. V-JEPA

- 位置：`/Users/jqwang/29-autoresearch/V-JEPA`
- 当前状态：本地先有研究文档和路线，不直接把完整训练仓并进来
- 进入位置：`infrastructure/perception/vjepa`

### 4. Qwen3.5-9B

- 模型位置：服务器侧离线模型目录
- 价值：任务解释、缺陷表述、结构化输出、研究 judge
- 进入位置：`infrastructure/language/qwen`

### 5. 质检业务案例

- 来源：`/Users/jqwang/22-开源版的teamviewer/Vuzix-M400/质量检测试点案例`
- 价值：定义 `session / stage / evidence / defect` 本体时的真实业务约束
- 进入位置：`docs/architecture`、`configs/data_recipes`

## 资产使用规则

- 第三方框架只通过 adapter 接入
- 业务案例先转成数据本体和评估标准，不直接硬编码进模型层
- 大数据和大权重不提交到 git，仓库里只保留 recipe、路径约定和运行脚本
