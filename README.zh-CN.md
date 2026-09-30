<div align="center">

# 数值实验管理器

面向可复现 Python 数值/ML 实验的指导型 skill：结构版本化、torch.fx 数据流、墙钟基线
对比、早停、小幅参数微调，以及用 skill 内置测试方程库诊断模型弱项。

[English](README.md) · [贡献者](CONTRIBUTORS.md) · [工作流](#工作流) · [脚本](#脚本) · [安装](#安装) · [许可证与边界](#许可证与边界)

![version](https://img.shields.io/badge/version-0.1.0-blue)
![scripts](https://img.shields.io/badge/scripts-5-2ea44f)
![license](https://img.shields.io/badge/license-MIT-green)

</div>

<p align="center">
  如果这个项目对你有帮助，欢迎为仓库点 Star ⭐
  <a href="https://github.com/zanew314/numerical-experiment-manager"><img alt="GitHub stars" src="https://img.shields.io/github/stars/zanew314/numerical-experiment-manager?style=social"></a>
</p>

## 这个仓库是什么

本仓库**本身就是这个 skill**。它教会编码 agent 把 Python 数值/ML 实验项目当作可复现的
战役来管理。指导主体在 [`SKILL.md`](SKILL.md)；五个轻量、零依赖的脚本只负责确定性记账。
它不是独立流水线——每一步交互都由 agent 在对话中执行。

## 工作流

| 工作流 | 触发 | 行为 |
| --- | --- | --- |
| A. 结构版本化 | 首次看到项目，或模型文件被修改 | 用 AST 生成版本化快照（`v0001`、`v0002`…）；作者的每次结构改动成为一个版本；出结果后把结构差异与指标交给 LLM 解释性能变化。 |
| B. 自动化测试 | 用户说“帮我测试”/“run this” | 运行程序，可选用 `torch.fx` 追踪数据流，产出精确结构图与实验报告；当相对误差长期高于 0.10 时早停；可微调少量小参数。 |
| C. 方程库诊断 | 用户问“模型哪里不行” | 从 skill 内置的 Markdown 方程库生成测试方程，按模型输入格式适配，每个方程少量短迭代运行，按族与难度轴聚合，产出诊断与**仅建议**的结构改进。 |

证据纪律：保留原始 loss 序列，在等墙钟下比较 **best-so-far**，绝不用看似合理的摘要替换
真实测量。

## 脚本

| 脚本 | 用途 |
| --- | --- |
| [`scripts/scan_structure.py`](scripts/scan_structure.py) | AST 扫描 → 版本化的 `.nems/structure/` JSON + Markdown。 |
| [`scripts/diff_structure.py`](scripts/diff_structure.py) | 比较两个版本的结构差异 → 供 LLM 使用的 JSON。 |
| [`scripts/fx_dataflow.py`](scripts/fx_dataflow.py) | `torch.fx` 追踪 → Mermaid/DOT 数据流图 + 形状。 |
| [`scripts/watch_experiment.py`](scripts/watch_experiment.py) | 运行程序、记录墙钟 loss、早停、重训一次并写运行产物。 |
| [`scripts/testset.py`](scripts/testset.py) | 列出方程卡、校验算例、将各算例运行聚合为分组摘要。 |

## 快速开始

```bash
git clone https://github.com/zanew314/numerical-experiment-manager.git
cd numerical-experiment-manager

# 1. 版本化项目结构（首次扫描生成 v0001）。
python scripts/scan_structure.py /path/to/project

# 2. 作者修改模型后再次扫描（生成 v0002）并比较。
python scripts/scan_structure.py /path/to/project
python scripts/diff_structure.py /path/to/project --from v0001 --to v0002

# 3. 追踪数据流（可选，需要 PyTorch）。
python scripts/fx_dataflow.py /path/to/project/model.py \
    --entry MLP --input-shape 1,2 --save .nems/figures

# 4. 先记录一次基线，再运行并监控实验。
python scripts/watch_experiment.py /path/to/project --run-id baseline \
    --cmd "python main.py" --loss-source jsonl
python scripts/watch_experiment.py /path/to/project --run-id run_001 \
    --cmd "python main.py" --baseline .nems/baseline/loss_time.jsonl \
    --tolerance 0.10 --sustain 3

# 5. 在方程库上诊断：列出卡、实例化算例、聚合结果。
python scripts/testset.py list
python scripts/testset.py scaffold poisson-2d --out .nems/testset/poisson-2d/case.json
python scripts/testset.py aggregate /path/to/project --testset .nems/testset
```

随后填写 [`prompts/structure_diff.md`](prompts/structure_diff.md) 与
[`prompts/experiment_report.md`](prompts/experiment_report.md)，由 agent 写出变更报告与
`REPORT.md`。

## 产物

所有持久状态都写在被管理项目根目录的 `.nems/` 下：

```text
.nems/
├── structure/   # index.json + vNNNN.json/md（版本化 AST 快照）
├── baseline/    # 固定的墙钟基线曲线
├── runs/<id>/   # loss_time.jsonl、metrics.json、early_stop.json、REPORT.md
├── testset/     # index.json、summary.json、<case_id>/case.json + 运行产物
├── reports/     # vNNNN_summary.md、*_change.md、diagnosis_<tag>.md
└── figures/     # dataflow_*.md/dot
```

`.nems/` 是证据而非源码，应加入 `.gitignore`。完整契约见
[`references/artifacts.md`](references/artifacts.md)。

## 安装

推荐使用 AI 辅助安装：让编码 agent 克隆本仓库、读取 Skill 说明、把仓库根目录链接到本地
skills 目录，并验证可发现性。

```text
请为我安装 Numerical Experiment Manager Skill。

仓库：https://github.com/zanew314/numerical-experiment-manager.git
分支：main

步骤：
1. 本地 clone 或更新仓库。
2. 读取 README.md 与 SKILL.md。
3. 如果当前环境支持本地 Skill discovery，把仓库根目录（含 SKILL.md 的目录）链接到本地 skills 目录。
4. 验证安装后的 Skill 可被发现。
5. 告诉我安装路径、是否需要重启，并给我一个测试 prompt。
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
New-Item -ItemType Junction `
  -Path "$env:USERPROFILE\.config\opencode\skill\numerical-experiment-manager" `
  -Target "E:\path\to\numerical-experiment-manager"
```

如果你的 agent 使用别的本地 Skill 目录，请替换为对应配置路径。

## 验证

无构建步骤。脚本仅使用 Python 标准库（`fx_dataflow.py` 额外需要 PyTorch）。在仓库根目录运行
冒烟测试：

```bash
python .github/smoke_test.py
```

## 许可证与边界

MIT（见 [`LICENSE`](LICENSE)）。不要提交生成的运行产物、私有数据集、`.env` 文件或密钥。
`.nems/` 是生成的证据，在被管理项目中应加入 `.gitignore`。本 skill 面向 Python 项目；
Julia 与 MATLAB 不在范围内。
