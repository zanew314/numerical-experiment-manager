<div align="center">

# 数值实验管理器

面向可复现 Python 数值/ML 实验的指导型 skill package：结构版本化、torch.fx 数据流、
墙钟基线对比、早停与小幅参数微调。

[English](README.md) · [贡献者](CONTRIBUTORS.md) · [技能包](#技能包) · [安装](#安装) · [快速开始](#快速开始) · [安全边界](#安全边界)

![version](https://img.shields.io/badge/version-0.1.0-blue)
![skills](https://img.shields.io/badge/skills-1-2ea44f)
![license](https://img.shields.io/badge/license-MIT-green)

</div>

<p align="center">
  如果这个项目对你有帮助，欢迎为仓库点 Star ⭐
  <a href="https://github.com/zanew314/numerical-experiment-manager"><img alt="GitHub stars" src="https://img.shields.io/github/stars/zanew314/numerical-experiment-manager?style=social"></a>
</p>

## 这个仓库是什么

本仓库是数值实验管理器的所在地。它为 Python 数值/ML 实验项目维护一份版本化的结构记录，
并在用户要求时进行测试：在固定的墙钟基线上运行程序、用 `torch.fx` 追踪模型数据流、生成
实验报告、在疑似落入局部最优时终止运行，并对小参数做微调。

设计以**指导为核心**：工作流写在 package 的 `SKILL.md` 与 `prompts/` 中；少量轻量、零依赖
的脚本只负责确定性记账，交互步骤全部由编码 agent 驱动。根页面作为公开地图，具体使用请
进入 skill package。

## 技能包

| 包 | 用途 | 入口 |
| --- | --- | --- |
| [`numerical-experiment-manager`](skills/numerical-experiment-manager/) | 结构版本化与差异比较、torch.fx 数据流、固定基线的等墙钟实验、局部最优早停、小参数微调。 | [`README`](skills/numerical-experiment-manager/README.zh-CN.md) · [`SKILL`](skills/numerical-experiment-manager/SKILL.md) |

## 安装

推荐使用 AI 辅助安装：让编码 agent 克隆或更新本仓库、读取 Skill 说明、链接包含 `SKILL.md`
的 package，并验证可发现性。

```text
请为我安装 Numerical Experiment Manager Skill。

仓库：https://github.com/zanew314/numerical-experiment-manager.git
分支：main
Skill 路径：
- skills/numerical-experiment-manager

步骤：
1. 本地 clone 或更新仓库。
2. 读取 README.md、skills/numerical-experiment-manager/SKILL.md，以及 AGENTS.md（如果存在）。
3. 如果当前环境支持本地 Skill discovery，把包含 SKILL.md 的目录链接到本地 skills 目录。
4. 验证安装后的 Skill 可被发现。
5. 告诉我安装路径、是否需要重启，并给我一个测试 prompt。
```

opencode 与 Codex 风格的本地 discovery 手动回退：

```bash
git clone https://github.com/zanew314/numerical-experiment-manager.git
cd numerical-experiment-manager

# opencode skill 目录
mkdir -p ~/.config/opencode/skill
ln -s "$PWD/skills/numerical-experiment-manager" \
      ~/.config/opencode/skill/numerical-experiment-manager

# Codex 风格本地 discovery
mkdir -p ~/.codex/skills
ln -s "$PWD/skills/numerical-experiment-manager" \
      ~/.codex/skills/numerical-experiment-manager
```

Windows 上用 junction 代替 symlink：

```powershell
New-Item -ItemType Junction `
  -Path "$env:USERPROFILE\.config\opencode\skill\numerical-experiment-manager" `
  -Target "E:\path\to\numerical-experiment-manager\skills\numerical-experiment-manager"
```

如果你的 agent 使用别的本地 Skill 目录，请替换为对应配置路径。

## 快速开始

```bash
git clone https://github.com/zanew314/numerical-experiment-manager.git
cd numerical-experiment-manager/skills/numerical-experiment-manager

python scripts/scan_structure.py /path/to/project
python scripts/diff_structure.py /path/to/project --from v0001 --to v0002
python scripts/fx_dataflow.py /path/to/project/model.py --entry MLP --input-shape 1,2
python scripts/watch_experiment.py /path/to/project --run-id baseline --cmd "python main.py"
```

完整工作流见 package 的 [README](skills/numerical-experiment-manager/README.zh-CN.md) 与
[`SKILL.md`](skills/numerical-experiment-manager/SKILL.md)。

## 仓库结构

```text
numerical-experiment-manager/
├── README.md
├── README.zh-CN.md
├── SKILL.md
├── CONTRIBUTORS.md
├── LICENSE
└── skills/
    └── numerical-experiment-manager/
        ├── README.md
        ├── README.zh-CN.md
        ├── SKILL.md
        ├── agents/
        ├── prompts/
        ├── references/
        ├── scripts/
        └── requirements-test.txt
```

每个 package 的 prompts、references、scripts 与生成证据都应放在该 package 内。

## 验证

本仓库无根级构建步骤。四个 package 脚本仅使用 Python 标准库；`fx_dataflow.py` 额外需要
PyTorch。要验证改动，可对一个小项目运行这些脚本（见 package README），并检查 package 的
`SKILL.md` 与 README 链接。

## 安全边界

不要提交生成的运行产物、私有数据集、`.env` 文件或密钥。`.nems/` 是生成的证据，在被管理
项目中应加入 `.gitignore`。公开示例应包含 benchmark 数据来源说明。本 skill 面向 Python
项目；Julia 与 MATLAB 不在范围内。
