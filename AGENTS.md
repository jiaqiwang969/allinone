# Allinone 仓库准则

本仓库采用 DDD 分层结构：

- `domain`：只放业务语义、规则和不变量
- `application`：只放用例编排
- `infrastructure`：接入 `YOLO / V-JEPA / Qwen / autoresearch`
- `interfaces`：CLI、API、作业入口
- `bootstrap`：依赖装配与环境绑定

关键要求：

1. `domain` 不得直接依赖 `torch`、`transformers`、`ultralytics` 等基础设施库
2. 模型能力只能通过 adapter 进入业务层
3. 第一阶段只围绕 `session -> guidance -> evidence -> research` 主闭环
