"""LLM code analysis for the experiment manager.

The coding agent is the primary "LLM": it reads ``prompts/structure_analysis.md``,
fills ``{{CODE_CONTENT}}`` and reasons about the project. This module supports that
path with:

1. an OpenAI-compatible remote call (used when an API key is configured), and
2. a deterministic offline analyzer used for demos/tests and as a fallback.

Neither path uses ``ast`` or dynamic probing. Extraction is text/regex based, so the
same code works on partially generated or injected sources.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
BOOL_NONE = {"True": True, "False": False, "None": None}
GET_PARAM_RE = re.compile(
    r"get_param\(\s*[\"']([\w]+)[\"']\s*,\s*default\s*=\s*([^\)]+?)\s*\)"
)

KEYWORDS = (
    "lr", "learning_rate", "learn_rate", "epoch", "batch", "hidden", "hid", "layer",
    "depth", "dim", "width", "size", "weight", "tol", "threshold", "eps", "seed",
    "momentum", "dropout", "alpha", "beta", "gamma", "step", "lambda", "tau",
    "penalty", "activation", "optimizer", "scheduler", "iter", "penalty", "scale",
)

UNCHANGED_FIELDS = ("architecture", "optimizer", "loss", "test_function", "training")


def parse_literal(text: str):
    """Parse a Python literal into ``(value, type_name)`` or ``None``."""
    t = text.strip().rstrip(",")
    if not t:
        return None
    gp = GET_PARAM_RE.search(t)
    if gp:
        inner = parse_literal(gp.group(2))
        if inner:
            return inner
        return None
    if NUMBER_RE.match(t):
        if re.search(r"[.eE]", t):
            try:
                return float(t), "float"
            except ValueError:
                return None
        try:
            return int(t), "int"
        except ValueError:
            return None
    if t in BOOL_NONE:
        v = BOOL_NONE[t]
        return v, ("null" if v is None else "bool")
    m = re.fullmatch(r"([\"'])(.*)\1", t)
    if m:
        return m.group(2), "str"
    return None


COUNTER_SUFFIXES = ("_done", "_count", "_counter", "_idx", "_index", "_len", "_path", "_dir")


def _is_param_name(name: str, strict: bool = False) -> bool:
    lower = name.lower()
    if any(lower.endswith(suffix) for suffix in COUNTER_SUFFIXES):
        return False
    if any(k in lower for k in KEYWORDS):
        return True
    if not strict and name.isupper() and len(name) >= 2:
        return True
    return False


def _record(results: List[dict], seen: Dict[str, int], entry: dict) -> None:
    name = entry["name"]
    if name in seen:
        existing = results[seen[name]]
        rank = {"high": 3, "medium": 2, "low": 1}
        if rank.get(entry["confidence"], 0) > rank.get(existing["confidence"], 0):
            results[seen[name]] = entry
        return
    seen[name] = len(results)
    results.append(entry)


def extract_hyperparameters(files: Dict[str, str]) -> List[dict]:
    """Extract candidate hyperparameters from a ``rel_path -> source`` mapping."""
    results: List[dict] = []
    seen: Dict[str, int] = {}
    attr_re = re.compile(
        r"^(\s*)([A-Za-z_]\w*)\s*=\s*([^#\n]+?)\s*(?:#.*)?$"
    )
    self_re = re.compile(
        r"^(\s*)self\.([A-Za-z_]\w*)\s*=\s*([^#\n]+?)\s*(?:#.*)?$"
    )
    class_re = re.compile(r"^(\s*)class\s+(\w+)")
    arg_re = re.compile(r"add_argument\(\s*[\"']--?([\w\-]+)[\"'](.*)")

    for rel, content in files.items():
        lines = content.splitlines()
        current_class = None
        class_indent = -1
        for i, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            m = class_re.match(line)
            if m:
                current_class = m.group(2)
                class_indent = len(m.group(1))
                continue
            if current_class is not None:
                indent = len(line) - len(line.lstrip())
                if indent <= class_indent and re.match(r"\s*(def|class)\s", line):
                    current_class = None

            m = attr_re.match(line)
            if m:
                name, raw = m.group(2), m.group(3).strip()
                parsed = parse_literal(raw)
                indent = len(m.group(1))
                in_class = bool(current_class) and indent > class_indent
                is_upper = name.isupper()
                if parsed and (is_upper or _is_param_name(name)):
                    if in_class and is_upper:
                        conf, form = "high", "class_attr"
                    elif not in_class and is_upper:
                        conf, form = "high", "module_const"
                    else:
                        conf, form = "medium", "module_const"
                    cls = current_class if in_class else None
                    loc = f"{rel}:{i}" + (f" ({cls}.{name})" if cls else "")
                    _record(results, seen, {
                        "name": name, "value": parsed[0], "type": parsed[1],
                        "location": loc, "confidence": conf, "form": form,
                        "class_name": cls, "file": rel, "line": i, "raw": raw,
                    })
                    continue

            m = self_re.match(line)
            if m:
                name, raw = m.group(2), m.group(3).strip()
                parsed = parse_literal(raw)
                if parsed and _is_param_name(name, strict=True):
                    cls = current_class or ""
                    loc = f"{rel}:{i} ({cls + '.' if cls else ''}{name})"
                    _record(results, seen, {
                        "name": name, "value": parsed[0], "type": parsed[1],
                        "location": loc, "confidence": "medium", "form": "self_attr",
                        "class_name": cls or None, "file": rel, "line": i, "raw": raw,
                    })
                    continue

            m = arg_re.search(line)
            if m:
                name = m.group(1).replace("-", "_")
                dm = re.search(r"default\s*=\s*([^,\)]+)", m.group(2))
                if dm:
                    parsed = parse_literal(dm.group(1))
                    if parsed:
                        _record(results, seen, {
                            "name": name, "value": parsed[0], "type": parsed[1],
                            "location": f"{rel}:{i} (argparse --{name})",
                            "confidence": "medium", "form": "argparse",
                            "class_name": None, "file": rel, "line": i,
                            "raw": dm.group(1).strip(),
                        })
    return results


class LLMAnalyzer:
    """Produce structure / diff / report JSON via LLM or offline fallback."""

    def __init__(
        self,
        prompts_dir=None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ):
        self.prompts_dir = Path(prompts_dir) if prompts_dir else DEFAULT_PROMPTS_DIR
        self.api_key = api_key or os.environ.get("NEMS_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("NEMS_LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("NEMS_LLM_MODEL") or "gpt-4o-mini"
        self.timeout = timeout

    # ------------------------------------------------------------------ prompts
    def load_prompt(self, name: str) -> str:
        return (self.prompts_dir / name).read_text(encoding="utf-8")

    @staticmethod
    def render(template: str, mapping: Dict[str, str]) -> str:
        out = template
        for key, value in mapping.items():
            out = out.replace("{{" + key + "}}", value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))
        return out

    def build_structure_prompt(self, files: Dict[str, str]) -> str:
        code = self._join_code(files)
        return self.render(self.load_prompt("structure_analysis.md"), {"CODE_CONTENT": code})

    def build_diff_prompt(self, old_summary, changed_files, unchanged_files, changed_code, snapshot_id) -> str:
        template = self.load_prompt("diff_analysis.md")
        template = template.replace("{{SNAPSHOT_ID}}", str(snapshot_id))
        return self.render(template, {
            "OLD_SUMMARY": old_summary,
            "CHANGED_FILES": changed_files,
            "UNCHANGED_FILES": unchanged_files,
            "CHANGED_CODE": self._join_code(changed_code),
        })

    def build_report_prompt(self, experiment, baseline, history, aligned) -> str:
        return self.render(self.load_prompt("report_analysis.md"), {
            "EXPERIMENT": experiment,
            "BASELINE": baseline,
            "HISTORY": history,
            "ALIGNED_COMPARISON": aligned,
        })

    @staticmethod
    def _join_code(files: Dict[str, str]) -> str:
        parts = []
        for rel, content in files.items():
            parts.append(f"# ===== {rel} =====\n{content}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------- remote
    def analyze_structure(self, files: Dict[str, str]) -> dict:
        if self.api_key:
            try:
                raw = self._remote_chat(self.build_structure_prompt(files))
                data = parse_json(raw)
                if isinstance(data, dict) and "hyperparameters" in data:
                    data.setdefault("_source", "remote_llm")
                    return data
            except Exception:
                pass
        data = self._offline_structure(files)
        data["_source"] = "offline_fallback"
        return data

    def _remote_chat(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------ offline
    def analyze_diff(self, old_summary, changed_files, unchanged_files, changed_code, snapshot_id) -> dict:
        if self.api_key:
            try:
                raw = self._remote_chat(self.build_diff_prompt(
                    old_summary, changed_files, unchanged_files, changed_code, snapshot_id))
                data = parse_json(raw)
                if isinstance(data, dict) and "changed_summary" in data:
                    return data
            except Exception:
                pass
        return self._offline_diff(old_summary, changed_files, changed_code, snapshot_id)

    def analyze_report(self, experiment, baseline, history, aligned) -> dict:
        if self.api_key:
            try:
                raw = self._remote_chat(self.build_report_prompt(
                    experiment, baseline, history, aligned))
                data = parse_json(raw)
                if isinstance(data, dict) and "verdict" in data:
                    return data
            except Exception:
                pass
        return self._offline_report(experiment, baseline, aligned)

    # ------------------------------------------------------ offline: structure
    def _offline_structure(self, files: Dict[str, str]) -> dict:
        text = "\n".join(files.values())
        hparams = extract_hyperparameters(files)
        hmap = {h["name"]: h["value"] for h in hparams}
        model_class = self._model_class(files)
        architecture = self._parse_architecture(files, hmap)
        return {
            "model_type": self._model_type(text, architecture),
            "architecture": architecture,
            "optimizer": self._parse_optimizer(text, hmap),
            "loss": self._parse_loss(text),
            "test_function": self._parse_test_function(text),
            "data_flow": self._build_data_flow(files, architecture, hmap, model_class),
            "training": self._parse_training(files, hmap),
            "entry_point": self._entry_point(files),
            "hyperparameters": hparams,
        }

    @staticmethod
    def _linear_layers(files: Dict[str, str]):
        pattern = re.compile(
            r"self\.(\w+)\s*=\s*nn\.Linear\(\s*([^,]+?)\s*,\s*([^,\)]+?)\s*\)"
        )
        layers = []
        for rel, content in files.items():
            for m in pattern.finditer(content):
                layers.append({
                    "name": m.group(1),
                    "in": m.group(2).strip(),
                    "out": m.group(3).strip(),
                    "file": rel,
                    "line": content[:m.start()].count("\n") + 1,
                })
        return layers

    @staticmethod
    def _resolve_dim(token: str, hmap: Dict[str, object]):
        token = token.strip()
        parsed = parse_literal(token)
        if parsed:
            return parsed[0]
        bare = re.sub(r"^(?:self\.)?(?:config|cfg|args|params|opt)\.", "", token)
        if bare in hmap:
            return hmap[bare]
        if token in hmap:
            return hmap[token]
        return token

    @staticmethod
    def _model_type(text: str, architecture: dict) -> str:
        if re.search(r"PINN|physics[-_ ]?informed", text, re.IGNORECASE):
            return "PINN"
        if "ParticleWNN" in text:
            return "ParticleWNN"
        if re.search(r"Deep\s*Ritz|ritz", text, re.IGNORECASE):
            return "Deep Ritz"
        if re.search(r"\bFNO\b|Fourier", text):
            return "FNO"
        if re.search(r"\bWAN\b", text):
            return "WAN"
        if architecture.get("layers"):
            return "MLP"
        return "其他"

    def _parse_architecture(self, files, hmap) -> dict:
        layers = self._linear_layers(files)
        text = "\n".join(files.values())
        if re.search(r"\bTanh\b|torch\.tanh|F\.tanh", text):
            activation = "tanh"
        elif re.search(r"\bReLU\b|torch\.relu|F\.relu", text):
            activation = "relu"
        elif re.search(r"\bSigmoid\b|torch\.sigmoid", text):
            activation = "sigmoid"
        elif re.search(r"\bGELU\b|F\.gelu", text):
            activation = "gelu"
        else:
            activation = None
        init_m = re.search(r"xavier\w*|kaiming\w*|orthogonal_|uniform_|normal_", text)
        return {
            "layers": len(layers),
            "neurons_per_layer": [self._resolve_dim(l["out"], hmap) for l in layers],
            "activation": activation,
            "initialization": init_m.group(0) if init_m else None,
        }

    def _parse_optimizer(self, text, hmap) -> dict:
        opt_m = re.search(r"torch\.optim\.(\w+)", text)
        sched_m = re.search(
            r"(StepLR|ExponentialLR|CosineAnnealingLR|MultiStepLR|ReduceLROnPlateau|OneCycleLR)",
            text,
        )
        lr = hmap.get("LR", hmap.get("lr", hmap.get("learning_rate")))
        if lr is None:
            lr_m = re.search(r"lr\s*=\s*([^,\)\n]+)", text)
            if lr_m:
                parsed = parse_literal(lr_m.group(1))
                lr = parsed[0] if parsed else lr_m.group(1).strip()
        if re.search(r"ADMM", text, re.IGNORECASE):
            special = "ADMM"
        elif re.search(r"adversarial|对抗", text, re.IGNORECASE):
            special = "对抗训练"
        elif re.search(r"alternat|交替", text, re.IGNORECASE):
            special = "交替方向法"
        else:
            special = "无"
        return {
            "type": opt_m.group(1) if opt_m else None,
            "lr": lr,
            "scheduler": sched_m.group(1) if sched_m else None,
            "special": special,
        }

    @staticmethod
    def _parse_loss(text) -> dict:
        formulas = {
            "MSELoss": ("mse", "mean((y_pred - y_true)^2)"),
            "L1Loss": ("l1", "mean(|y_pred - y_true|)"),
            "CrossEntropyLoss": ("cross_entropy", "-mean(log softmax(y_pred)[y_true])"),
            "HuberLoss": ("huber", "huber(y_pred, y_true)"),
        }
        items = []
        for cls, (name, formula) in formulas.items():
            if cls in text:
                items.append({"name": name, "weight": 1.0, "formula": formula})
        return {"terms": items} if items else {"terms": [{"name": "custom", "weight": 1.0, "formula": "unknown"}]}

    @staticmethod
    def _parse_test_function(text):
        if re.search(r"\bsin\b|math\.sin|torch\.sin|np\.sin", text):
            return "sin(x)"
        if re.search(r"\bcos\b|math\.cos|torch\.cos|np\.cos", text):
            return "cos(x)"
        if re.search(r"math\.exp|torch\.exp|np\.exp", text):
            return "exp(x)"
        return None

    @staticmethod
    def _parse_training(files, hmap) -> dict:
        text = "\n".join(files.values())
        epochs = hmap.get("EPOCHS", hmap.get("epochs", hmap.get("num_epochs")))
        if epochs is None:
            m = re.search(r"range\(\s*([^\)]+)\)", text)
            epochs = m.group(1).strip() if m else None
        batch = hmap.get("BATCH_SIZE", hmap.get("batch_size"))
        if batch is None:
            m = re.search(r"batch_size\s*=\s*([^,\)]+)", text)
            batch = m.group(1).strip() if m else "full-batch"
        if re.search(r"linspace|arange", text):
            sampling = "grid"
        elif re.search(r"randn|torch\.rand|np\.random", text):
            sampling = "random"
        else:
            sampling = "unknown"
        return {"epochs": epochs, "batch_size": batch, "sampling_strategy": sampling}

    @staticmethod
    def _entry_point(files) -> str:
        for rel, content in files.items():
            if "__main__" in content:
                return rel
        return next(iter(files), "main.py")

    @staticmethod
    def _model_class(files) -> str:
        for content in files.values():
            lines = content.splitlines()
            current = None
            for line in lines:
                m = re.match(r"\s*class\s+(\w+)", line)
                if m:
                    current = m.group(1)
                if current and "nn.Linear(" in line:
                    return current
        return "MLP"

    def _build_data_flow(self, files, architecture, hmap, model_class) -> List[dict]:
        layers = self._linear_layers(files)
        data_node = "main.py:make_data"
        for rel, content in files.items():
            if re.search(r"linspace|randn|torch\.rand|make_|load_data", content):
                data_node = f"{rel}:make_data"
                break
        if not layers:
            return []
        flow: List[dict] = []
        first = layers[0]["name"]
        in_dim = self._resolve_dim(layers[0]["in"], hmap)
        flow.append({"from": data_node, "to": f"{model_class}.{first}", "content": f"x: [N, {in_dim}]"})
        flow.append({"from": data_node, "to": "loss", "content": "y: [N, 1]"})
        for idx in range(1, len(layers)):
            prev, cur = layers[idx - 1], layers[idx]
            dim = self._resolve_dim(prev["out"], hmap)
            flow.append({
                "from": f"{model_class}.{prev['name']}",
                "to": f"{model_class}.{cur['name']}",
                "content": f"h{idx}: [batch, {dim}]",
            })
        last = layers[-1]
        out_dim = self._resolve_dim(last["out"], hmap)
        flow.append({
            "from": f"{model_class}.{last['name']}",
            "to": "loss",
            "content": f"y_pred: [batch, {out_dim}]",
        })
        return flow

    # ----------------------------------------------------------- offline: diff
    def _offline_diff(self, old_summary, changed_files, changed_code, snapshot_id) -> dict:
        new = self._offline_structure(changed_code)
        changes = []
        for key in ("model_type", "architecture", "optimizer", "loss", "test_function", "training", "entry_point"):
            before = old_summary.get(key) if isinstance(old_summary, dict) else None
            after = new.get(key)
            if before != after:
                changes.append({
                    "field": key,
                    "before": before,
                    "after": after,
                    "reason": "changed source file: " + ", ".join(changed_files),
                })
        changed_field_names = {c["field"] for c in changes}
        unchanged = {
            key: f"unchanged (see {snapshot_id})"
            for key in UNCHANGED_FIELDS
            if key not in changed_field_names
        }
        old_hp = {}
        if isinstance(old_summary, dict):
            for h in old_summary.get("hyperparameters", []):
                old_hp[h.get("name")] = h.get("value")
        new_hp = {h["name"]: h["value"] for h in new.get("hyperparameters", [])}
        hp_changes = []
        for name, value in new_hp.items():
            if name in old_hp and old_hp[name] != value:
                hp_changes.append({"name": name, "old": old_hp[name], "new": value, "location": changed_files[0] if changed_files else ""})
        return {
            "changed_summary": new,
            "unchanged": unchanged,
            "changes": changes,
            "hyperparameter_changes": hp_changes,
        }

    # --------------------------------------------------------- offline: report
    def _offline_report(self, experiment, baseline, aligned) -> dict:
        ratios = [e["ratio"] for e in aligned if e.get("ratio") is not None]
        finite = [r for r in ratios if r == r and r not in (float("inf"), float("-inf"))]
        early = experiment.get("early_stop") or {}
        verdict = "similar"
        reasons: List[str] = []
        if early.get("triggered"):
            verdict = "failed"
            reasons.append(
                f"在 wall-clock {early.get('trigger_second')} 秒时相对误差倍率达到 "
                f"{early.get('trigger_ratio'):.2f}，超过 1.2 阈值，判定为训练发散。"
            )
            reasons.append("最可能原因是学习率过大，Adam 更新步长超出稳定范围。")
        elif finite and max(finite) > 1.1:
            verdict = "worse"
            reasons.append("相同 wall-clock 时间点下相对误差高于基线，说明该配置收敛更慢。")
        elif finite and max(finite) < 0.95:
            verdict = "improved"
            reasons.append("相同 wall-clock 时间点下相对误差低于基线，说明该配置收敛更快。")
        else:
            reasons.append("相同 wall-clock 时间点下相对误差与基线接近，差异不显著。")
        recs = []
        if verdict == "improved":
            recs.append("建议保留该配置，并可进一步微调相近的学习率。")
        elif verdict == "failed":
            recs.append("建议调低学习率后重新实验。")
        else:
            recs.append("可尝试微调隐藏层宽度或层数。")
        return {
            "verdict": verdict,
            "summary": reasons[0],
            "evidence": aligned,
            "possible_reasons": reasons,
            "recommendations": recs,
            "early_stop_analysis": early if early.get("triggered") else None,
        }


def parse_json(text: str) -> dict:
    """Best-effort JSON extraction from an LLM response."""
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return {}
