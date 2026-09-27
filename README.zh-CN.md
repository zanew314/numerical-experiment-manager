# AI 数值实验管理器 Skill

English guide: [README.md](README.md)

这个 skill 让编码 agent 管理 Python 数值/机器学习实验项目的全生命周期。
agent 只扫描项目一次，注入已确认的超参数，在固定基线上做等墙钟实验对比，及早停止
劣质实验，生成报告，并对小参数做微调。它是
[VeryMath](https://github.com/VeryMath) 组织下的一个独立项目。

## 适用范围

使用本 skill 处理能够记录“每个 epoch 的墙钟 loss”的 Python 项目。交互步骤由 agent
在对话中完成；`agent/` 下的模块是 agent 调用的确定性工具，不是独立 CLI。

支持：

- 一次性的 LLM 结构摘要，以及模型更新后的增量 diff；
- 不复制函数体即可保存模型结构的快照（AST 骨架 + 可选 torch.fx 图）；
- 超参数的确认、注入与结构化历史；
- 按相同经过秒数对齐的墙钟基线对比；
- 早停，以及至多一次自动重训；
- 针对不写 `loss_time.jsonl` 的程序的 loss 流适配器；
- 单实验与整体报告、`INDEX.md` 与演化图；
- 带保留/回退结论的小参数自动微调。

不要把 Julia、MATLAB、非 Python 或纯文档任务路由到本 package。被管理的程序应写出
`loss_time.jsonl`；若不能，则用 loss 流适配器读取它运行时的输出（见下文）。

## 依赖与隔离安装

- Python 3.10+
- agent 工具需要 `numpy`；demo 需要 `torch`（CPU）与 `matplotlib`
- 无外部 LLM SDK：远程调用使用 `urllib`，没有 API key 时分析器回退到确定性离线模式

创建隔离环境并安装测试依赖：

```bash
python3 -m venv .venv-nems
source .venv-nems/bin/activate
python -m pip install -r requirements-test.txt
```

修改已有环境前先取得批准。

## 快速开始

在本 package 目录运行自包含、仅 CPU 的演示（远低于三分钟）：

```bash
python examples/demo_mlp_sin/run_demo.py
```

它把 `main.py` 复制到 `examples/demo_mlp_sin/.demo_workspace` 沙箱，依次走完 Phase 0
到 Phase 8，并打印每个产物的写入位置。可选参数：

```bash
python examples/demo_mlp_sin/run_demo.py --interactive   # 交互选择保留/排除
python examples/demo_mlp_sin/run_demo.py --with-tuning   # 额外运行 Phase 7 微调
python examples/demo_mlp_sin/run_demo.py --workspace D:\tmp\nems_demo
```

预期结果：

- `exp_001`（LR=0.01）完成，并相对基线 **improved**；
- `exp_002_earlystop`（LR=10.0）在单次重训也失败后被判为 **early stopped** 并记录；
- 模型更新只重读变化的 `main.py`，写出 `project_summary_diff.md`，未变字段标注
  `unchanged (see exp_001)`。

demo 每次运行都会重建沙箱，因此重复运行会覆盖 `.demo_workspace`（已 gitignore，不提交
任何运行产物）。

第二个更短的 demo 管理一个**没有** `loss_time.jsonl`、只打印进度的项目：

```bash
python examples/demo_no_stream/run_demo.py
```

它会写出 `.nems/model_structure.json`/`.md`、供 `stdout_regex` 源使用的
`.nems/stream_adapter.json`、基线、一次正常实验与一次早停实验，以及常规报告。

一个科学算例用三种方法求解二维 Poisson 问题 `-Delta u = f`（精确解
`u* = sin(freq x1) sin(freq x2)`，`freq = pi`）：`main.py` 是 PINN，
`particle_wnn.py` 是弱形式 ParticleWNN，`traditional_solver.py` 是传统 5 点有限差分解法：

```bash
python examples/demo_poisson/run_demo.py
python examples/demo_poisson/traditional_solver.py --n 127
python examples/demo_poisson/particle_wnn.py
```

有限差分参考解以二阶收敛（`n=127` 时 L2 相对误差约 `2e-4`）；管理器随后在固定
基线上运行 PINN 实验，demo 也会直接运行 ParticleWNN。详见
`examples/demo_poisson/PROBLEM.md`。

## 接入合同

被管理的程序读取 `NEMS_OUTPUT_DIR` 并写出：

- `loss_time.jsonl` — 每个 epoch 一行 JSON：
  `{"wall_clock": <sec>, "loss": <float>, "epoch": <int>}`；
- `metrics.json` — `{"final_loss": ..., "final_error": ...,
  "total_wall_clock": ..., "epochs": ...}`。

程序还应遵守：

- `NEMS_MAX_WALL_CLOCK` — 单次运行的软秒数上限；
- `NEMS_SEED`（默认 `0`）与 `NEMS_RETRAIN_ATTEMPT`（首次运行 `0`，重训为 `1`），
  使自动重训可以改变初始化。

agent 的早停器会用 `sqrt` 把记录的 `loss` 转成 RMSE 形式的误差，因此被管理程序必须
记录非负 loss。生成的适配模块读取 `.nems/current_config.json`，文件缺失时回退到源码
默认值，因此项目仍可独立运行。

## 模型结构快照

`agent/model_structure.py` 在不复制函数体的前提下保存“模型是什么”。它始终用标准库
AST 生成骨架（import、常量、类基类、方法签名、`self.<layer> = Module(...)` 赋值），
写入 `.nems/model_structure.json` 与 `.nems/model_structure.md`。当 PyTorch 可用且能提供
模型实例与示例输入时，`augment_with_torch_fx` 会附加带形状与参数量的算子图；追踪是
尽力而为，失败也不会影响骨架。`diff_model_structures` 比较两份快照，用于增量摘要。
详见 `references/model_structure_methods.md`。

## loss 流适配器

当程序不写 `loss_time.jsonl` 时，在 `.nems/stream_adapter.json` 里选择来源（或给
`ExperimentRunner.run` 传 `stream_spec=`）：

| type | 读取 | 实时 |
| --- | --- | --- |
| `jsonl`（默认） | `loss_time.jsonl` | 是 |
| `stdout_regex` | 被捕获的 stdout / `logs/run.log` | 是 |
| `sidecar` | 程序自己写的 CSV / JSON / JSONL | 是 |
| `completion_only` | 仅最终 `metrics.json` | 否 |

每个来源都产出同样的 `{"wall_clock", "loss", "epoch"}` 点，因此早停与报告无需改动。
当来源没有 `wall_clock` 分组时，时间轴按 `index * time_step` 合成，并标记
`synthetic_time`。用 `loss_stream.register_source` 注册自定义来源，或用
`{"type": "custom", "module": ..., "class": ...}` 指向磁盘上的实现。详见
`references/loss_stream_adapters.md`。

## Runner 合同与边界

- 精度指标为 RMSE；在每个整数墙钟秒取 **best-so-far** 误差，避免瞬态 loss 尖峰造成
  误判。
- 不足一秒仍按第 1 秒计算（向上取整）。
- 早停：预热 2 秒后 `ratio > 1.20`；误差非有限或出现灾难性倍率（`> 2.0`）时立即停止。
- 当实验的 epoch 进度明显落后于基线时，速度护栏跳过该秒的倍率判定，避免机器抖动被
  误认为劣质配置。
- 重训一次：因相对误差过大而停止的运行会先自动重训一次（新进程、`NEMS_SEED` 左移、
  相同或覆盖后的配置），仍不达标才记为 early-stopped。第一次尝试的产物保存在
  `experiments/<exp_id>/attempts/`；`max_retrains=0` 可关闭。
- 任何编辑前都会把原始源码快照到 `.nems/code_snapshots/<timestamp>/`；被排除的超参数
  保持原有字面值。
- 适配模块名会自适应以避免冲突：`nems_config.py` -> `nems_param_loader.py` ->
  `_nems_config.py`。
- 小参数微调默认至多 3 个参数、每个至多 2 个取值。

### LLM 配置

编码 agent 是主 LLM。若要让 Python 工具调用 OpenAI 兼容端点，设置：

```bash
NEMS_LLM_API_KEY=...
NEMS_LLM_BASE_URL=https://api.openai.com/v1   # 可选
NEMS_LLM_MODEL=gpt-4o-mini                    # 可选
```

没有 key 时，`LLMAnalyzer` 使用离线回退并设置 `_source = "offline_fallback"`。

## 命令行入口

两个确定性、轻依赖脚本支撑平台式检查与 agent 工作流：

```bash
python scripts/check_environment.py        # JSON：必需/可选依赖
python scripts/analyze_project.py <dir>    # JSON：离线结构 + 超参数
```

两者都向 stdout 输出单个 JSON 文档；`analyze_project.py` 输入非法时以退出码 `2` 向
stderr 输出 JSON 错误。

## 验证

在本 package 目录运行：

```bash
python -m unittest discover -s tests -p "test_*.py"
```

端到端 demo 测试为可选，会跑完整生命周期：

```bash
# PowerShell
$env:NEMS_RUN_INTEGRATION="1"; python -m unittest tests.test_demo_pipeline -v
# bash
NEMS_RUN_INTEGRATION=1 python -m unittest tests.test_demo_pipeline -v
```

被跳过的集成测试不能作为发布证据；公开发布前请运行可选测试。

## 注册

把本 package 目录加入 `opencode.json` 的 `skills.paths`：

```json
{
  "skills": {
    "paths": ["skills"]
  }
}
```

## 许可证与来源边界

本 package 采用 MIT license（见 `LICENSE`）。它是 VeryMath 组织下的独立项目。demo
依赖 PyTorch，但本 package 不打包、也不重新许可 PyTorch 代码。

二维 Poisson 算例（`examples/demo_poisson/`）的问题设定与 ParticleWNN 方法改编自
[`yaohua32/Physics-Driven-Deep-Learning-for-PDEs`](https://github.com/yaohua32/Physics-Driven-Deep-Learning-for-PDEs)
（Apache-2.0）。**未复制上游代码**：`main.py`、`particle_wnn.py`、`traditional_solver.py`
是对同一数学问题的独立重写。

公开发布时应如实记录贡献者署名；Poisson 算例的来源见上文。
