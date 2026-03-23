# Migration Map

## 总原则

旧工作区 `29-autoresearch` 是“素材堆栈”，新仓 `31-allinone` 是“产品主仓”。迁移不是复制目录，而是按 DDD 语义重组。

## 目录级映射

| 来源 | 去向 | 说明 |
| --- | --- | --- |
| `29-autoresearch/autoresearch` | `src/allinone/infrastructure/research/autoresearch` | 只迁移自动实验相关能力 |
| `29-autoresearch/ultralytics` | `src/allinone/infrastructure/perception/yolo` | 只接入检测能力，不把整个第三方仓塞进来 |
| `29-autoresearch/V-JEPA` | `src/allinone/infrastructure/perception/vjepa` | 当前以接口和 recipe 先占位，后续接入实际模型 |
| `29-autoresearch/docs/plans` | `docs/plans` | 迁移为新主仓的设计与实施依据 |
| `29-autoresearch/data` | `configs/data_recipes` + 未来 `data/` | 先沉淀数据 recipe，再决定实际下载与落盘布局 |
| 服务器模型目录 | `configs/model_recipes` | 本地只保存约定，不提交大模型权重 |

## 不直接迁移的内容

- 第三方仓的全部 `.git` 历史
- 大模型权重和数据本体
- 只用于临时分析的 notebook 输出

## 迁移完成的判断标准

不是“目录都复制过来了”，而是满足下面三个条件：

1. `allinone` 可以解释每一份旧资产为什么存在
2. 每个旧资产都能指向一个新的 bounded context
3. 未来服务器执行、训练、回放都只围绕 `allinone` 展开
