你是一个代码分析专家。请阅读以下 Python 项目代码，输出一份功能级结构概括 JSON。

要求输出以下字段：

1. model_type: 模型类型（PINN / ParticleWNN / Deep Ritz / FNO / WAN / MLP / 其他）
2. architecture:
   - layers: 层数（整数）
   - neurons_per_layer: 每层神经元数（列表）
   - activation: 激活函数名
   - initialization: 初始化方式（若有）
3. optimizer:
   - type: 优化器类型
   - lr: 学习率
   - scheduler: 学习率调度（若无则 null）
   - special: 特殊算法（ADMM / 对抗训练 / 交替方向法 / 无）
4. loss:
   - 各项名称、权重、数学形式
5. test_function: 测试函数（若是弱形式方法，否则 null）
6. data_flow: 数据流图（重点）
   - 每一项格式：{"from": "...", "to": "...", "content": "张量名: [形状]"}
   - 例如：{"from": "main.py:load_data", "to": "MLP.fc1", "content": "x: [N, 1]"}
   - 例如：{"from": "MLP.fc3", "to": "MLP.fc4", "content": "h3: [batch, 64]"}
   - 张量形状可以推断，不要求绝对精确
7. training: epochs, batch_size, sampling_strategy
8. entry_point: 主文件路径
9. hyperparameters: 超参数列表
   - 每个参数：{"name": "...", "value": ..., "type": "...", "location": "文件:行号 或 类.属性", "confidence": "high/medium/low"}

请严格输出 JSON，不要添加任何解释文字。

代码：
{{CODE_CONTENT}}
