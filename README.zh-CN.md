<div align="center">

# 数值实验管理器

面向 Python 数值与机器学习实验的可复现墙钟管理：一次扫描、确认后的超参数注入、
固定基线对比、至多一次重训的早停、报告与微调。

[English](README.md) · [贡献者](CONTRIBUTORS.md) · [能力](#能力) · [安装](#安装) · [快速开始](#快速开始) · [许可证与边界](#许可证与边界)

![version](https://img.shields.io/badge/version-0.1.0-blue)
![modules](https://img.shields.io/badge/modules-11-2ea44f)
![license](https://img.shields.io/badge/license-MIT-green)

</div>

<p align="center">
  如果这个项目对你有帮助，欢迎为仓库点 Star ⭐
  <a href="https://github.com/zanew314/numerical-experiment-manager"><img alt="GitHub stars" src="https://img.shields.io/github/stars/zanew314/numerical-experiment-manager?style=social"></a>
</p>

## 这个仓库是什么

本仓库是数值实验管理器（Numerical Experiment Manager）的所在地，一个面向
实验型 Python 项目的 agent 原生 skill package。它把编码 agent 变成可复现的实验
管理器：agent 只扫描项目一次，注入已确认的超参数，在固定基线上做等墙钟实验对比，
及早停止明显劣质的运行，生成报告，并对小参数做微调。

本 skill 是**agent 原生**的。交互步骤（确认超参数、批准实验、批准微调）由编码
agent 在对话中完成；`agent/` 下的 Python 模块是 agent 调用的确定性工具，负责备份、
注入、子进程执行、墙钟对齐、报告与历史维护，并不是独立 CLI。

把每一次记录下来的对比都当作证据。在相同墙钟时间下比较 **best-so-far** 误差，保留原始
`loss_time.jsonl`，绝不用一个看起来合理的摘要替换真实测量。

以下情况请路由到别处：项目不是 Python（Julia、MATLAB 不在范围内）；程序完全无法产出
可用的每次运行输出（没有 `loss_time.jsonl`、没有可解析的 stdout、没有 sidecar 文件）；
或任务只是文档、打包，没有实验要运行。

## 能力

| 能力 | 用途 | 入口 |
| --- | --- | --- |
| 项目扫描 | 一次性 LLM 结构摘要与候选超参数。 | `agent/code_reader.py`、`agent/llm_analyzer.py` |
| 模型结构 | 不复制函数体的 AST 骨架，可选附加 `torch.fx` 图。 | `agent/model_structure.py` |
| 注入 | 备份源码、确认参数并注入 `get_param(...)`。 | `agent/hyperparameter_injector.py` |
| loss 流 | 读取 `jsonl`、`stdout_regex`、`sidecar`、`completion_only` 或自定义来源。 | `agent/loss_stream.py` |
| 基线与运行 | 在固定基线上做等墙钟运行，支持一次重训。 | `agent/experiment_runner.py` |
| 早停 | 逐秒 best-so-far 比较与触发类型分类。 | `agent/early_stopper.py` |
| 分析 | 在相同墙钟下比较实验、基线与历史。 | `agent/result_analyzer.py` |
| 报告 | 写出 `REPORT.md`、`INDEX.md`、`BASELINE_REPORT.md` 与图表。 | `agent/report_generator.py` |
| 微调 | 至多调整 3 个小参数，给出保留/回退结论。 | `agent/param_tuner.py` |
| 历史 | 维护 `.nems/history.json` 与实验索引。 | `agent/history_manager.py` |

随附资源：

| 路径 | 内容 |
| --- | --- |
| [`SKILL.md`](SKILL.md) | 面向 agent 的工作流与批准规则。 |
| `prompts/` | agent 填充的 LLM 提示模板。 |
| `references/` | 早停、超参数、模型结构、loss 流方法说明。 |
| `scripts/` | 轻依赖的 JSON 命令行检查。 |
| `examples/` | 三个自包含的 CPU 演示。 |
| `tests/` | 不依赖 pytest 的 `unittest` 测试套件。 |

## 安装

推荐使用 AI 辅助安装：让编码 agent 克隆或更新本仓库、读取 Skill 说明、安装入口并
验证可发现性。

```text
请为我安装 Numerical Experiment Manager Skill。

仓库：https://github.com/zanew314/numerical-experiment-manager.git
分支：main
Skill 路径：
- .（仓库根目录包含 SKILL.md）

步骤：
1. 本地 clone 或更新仓库。
2. 读取 README.md、SKILL.md，以及 AGENTS.md（如果存在）。
3. 如果当前环境支持本地 Skill discovery，把包含 SKILL.md 的目录链接到本地 skills 目录。
4. 如果某个 Skill 依赖相邻的共享支持目录，请保留这些 sibling 目录。
5. 验证安装后的 Skill 可被发现。
6. 告诉我安装路径、是否需要重启，并给我一个测试 prompt。
```

opencode 与 Codex 风格的本地 discovery 手动回退：

```bash
git clone https://github.com/zanew314/numerical-experiment-manager.git
cd numerical-experiment-manager

# opencode skill 目录
mkdir -p ~/.config/opencode/skill
ln -s "$PWD" ~/.config/opencode/skill/numerical-experiment-manager

# Codex 风格本地 discovery
mkdir -p ~/.codex/skills
ln -s "$PWD" ~/.codex/skills/numerical-experiment-manager
```

Windows 上用 junction 代替 symlink：

```powershell
New-Item -ItemType Junction -Path "$env:USERPROFILE\.config\opencode\skill\numerical-experiment-manager" -Target "E:\path\to\numerical-experiment-manager"
```

如果你的 agent 使用别的本地 Skill 目录，请把上面的路径换成对应配置路径。若要通过
`opencode.json` 注册，把本目录加入 `skills.paths`：

```json
{
  "skills": {
    "paths": ["skills"]
  }
}
```

运行演示前，创建隔离环境并安装测试依赖：

```bash
python3 -m venv .venv-nems
source .venv-nems/bin/activate
python -m pip install -r requirements-test.txt
```

修改已有环境前先取得批准。

## 快速开始

克隆仓库并运行自包含、仅 CPU 的演示（远低于三分钟）：

```bash
git clone https://github.com/zanew314/numerical-experiment-manager.git
cd numerical-experiment-manager
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
- `exp_002_earlystop`（LR=10.0）在单次重训也失败后被判为 **early stopped**；
- 模型更新只重读变化的 `main.py`，写出 `project_summary_diff.md`，未变字段标注
  `unchanged (see exp_001)`。

第二个更短的 demo 管理一个**没有** `loss_time.jsonl`、只打印进度的项目：

```bash
python examples/demo_no_stream/run_demo.py
```

一个科学算例用三种方法求解二维 Poisson 问题 `-Delta u = f`（精确解
`u* = sin(freq x1) sin(freq x2)`，`freq = pi`）：`main.py` 是 PINN，
`particle_wnn.py` 是弱形式 ParticleWNN，`traditional_solver.py` 是传统 5 点有限差分解法：

```bash
python examples/demo_poisson/run_demo.py
python examples/demo_poisson/traditional_solver.py --n 127
python examples/demo_poisson/particle_wnn.py
```

有限差分参考解以二阶收敛（`n=127` 时 L2 相对误差约 `2e-4`）；管理器随后在固定基线上
运行 PINN 实验。详见 `examples/demo_poisson/PROBLEM.md`。

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

当程序不写 `loss_time.jsonl` 时，在 `.nems/stream_adapter.json` 里选择 loss 来源
（或给 `ExperimentRunner.run` 传 `stream_spec=`）：

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

## 仓库结构

```text
numerical-experiment-manager/
├── README.md
├── README.zh-CN.md
├── SKILL.md
├── CONTRIBUTORS.md
├── LICENSE
├── pyproject.toml
├── requirements-test.txt
├── agent/
├── prompts/
├── references/
├── scripts/
├── examples/
│   ├── demo_mlp_sin/
│   ├── demo_no_stream/
│   └── demo_poisson/
└── tests/
```

请把演示、生成的运行证据与 `.nems/` 状态放在所属项目内；运行产物已 gitignore，不提交。

## 验证

本仓库无根级构建步骤。在本 package 目录运行：

```bash
python -m unittest discover -s tests -p "test_*.py"
```

端到端 demo 测试为可选，会跑完整生命周期：

```powershell
# PowerShell
$env:NEMS_RUN_INTEGRATION="1"; python -m unittest tests.test_demo_pipeline -v
```

```bash
# bash
NEMS_RUN_INTEGRATION=1 python -m unittest tests.test_demo_pipeline -v
```

`Validate skill` GitHub Actions 工作流在 Python 3.10 与 3.13 上运行零跳过的轻依赖测试，
随后用 CPU PyTorch 跑完整套件。被跳过的集成测试不能作为发布证据；公开发布前请运行
可选测试。

## 许可证与边界

本 package 采用 MIT License（见 `LICENSE`）。不要提交求解器 license、API key、私有
数据集、`.env` 文件、含敏感数据的生成日志或本地运行输出。公开示例应包含 benchmark
数据来源说明，并确保可以再分发。

demo 依赖 PyTorch，但本 package 不打包、也不重新许可 PyTorch 代码。二维 Poisson 算例
（`examples/demo_poisson/`）的问题设定与 ParticleWNN 方法改编自
[`yaohua32/Physics-Driven-Deep-Learning-for-PDEs`](https://github.com/yaohua32/Physics-Driven-Deep-Learning-for-PDEs)
（Apache-2.0）。**未复制上游代码**：`main.py`、`particle_wnn.py`、`traditional_solver.py`
是对同一数学问题的独立重写。署名记录在 [`CONTRIBUTORS.md`](CONTRIBUTORS.md)；公开发布时
应如实记录贡献者。
