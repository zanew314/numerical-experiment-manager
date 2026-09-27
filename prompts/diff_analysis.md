你是一个代码增量分析专家。项目此前已经生成过一份结构概括，现在只有部分文件发生了变更。
请只根据「变更文件的代码」和「变更文件的旧概括」更新结构概括，不要重新描述未变更的部分。

输入：

【旧的结构概括 JSON】
{{OLD_SUMMARY}}

【发生变更的文件列表】
{{CHANGED_FILES}}

【未发生变更的文件列表】
{{UNCHANGED_FILES}}

【变更文件的最新代码】
{{CHANGED_CODE}}

要求输出严格 JSON，包含：

1. changed_summary: 仅针对变更文件的更新后概括，字段与结构概括一致
   （model_type, architecture, optimizer, loss, test_function, data_flow, training,
   entry_point, hyperparameters）
2. unchanged: 未变更的部分，值为字符串 "unchanged (see {{SNAPSHOT_ID}})"，键名沿用
   结构概括中的字段名（例如 "optimizer": "unchanged (see exp_001)"）
3. changes: 列表，每一项 {"field": "...", "before": ..., "after": ..., "reason": "..."}
4. hyperparameter_changes: 列表，每一项 {"name": "...", "old": ..., "new": ..., "location": "..."}

不要输出任何解释文字，只输出 JSON。
