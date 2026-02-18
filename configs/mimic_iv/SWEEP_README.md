# Weights & Biases Sweep 使用指南

本指南说明如何使用 Weights & Biases (wandb) 进行超参数扫描。

## 安装依赖

首先安装 wandb：

```bash
pip install wandb
```

然后登录 wandb（首次使用需要）：

```bash
wandb login
```

## 配置文件说明

配置文件位于 `configs/` 目录下：

- `configs/sweep_config.yaml` - 完整的超参数扫描配置（包含所有可扫描的超参数）
- `configs/sweep_config_simple.yaml` - 简化版配置（只扫描关键超参数，适合快速实验）

### 主要超参数类别

#### 1. 训练超参数
- `learning_rate`: 学习率 (log_uniform: 0.0001 - 0.01)
- `batch_size`: 批次大小 (64, 128, 256)
- `weight_decay`: 权重衰减 (log_uniform: 1e-6 - 1e-3)
- `epochs`: 训练轮数 (固定为 100，可根据需要调整)

#### 2. MLP 模型架构超参数
- `mlp_hidden_dim_1/2/3`: MLP 隐藏层维度
- `mlp_latent_dim`: MLP 潜在空间维度
- `mlp_dropout`: MLP dropout 率

#### 3. 编码器超参数（用于预训练模型）
- `encoder_latent_dim`: 编码器潜在空间维度
- `encoder_hidden_dim_1/2`: 编码器隐藏层维度
- `encoder_dropout`: 编码器 dropout 率

#### 4. TabM 特定超参数
- `tabm_k_heads`: 集成头数
- `tabm_dropout`: TabM dropout 率

#### 5. TabNet 特定超参数
- `tabnet_n_d`: 决策维度
- `tabnet_n_a`: 注意力维度
- `tabnet_n_steps`: 决策步数
- `tabnet_gamma`: 特征重用惩罚系数

#### 6. VAE 超参数
- `vae_beta`: VAE KL 散度权重

#### 7. WAE 超参数
- `wae_lambda_ot`: 最优传输损失权重
- `wae_sinkhorn_eps`: Sinkhorn 正则化参数

#### 8. GW 超参数
- `gw_weight`: Gromov-Wasserstein 损失权重

#### 9. 其他超参数
- `use_log_ratio`: 是否使用 log ratio 损失

## 使用方法

### 在 JupyterLab 中运行（推荐）

#### 方法 1: 使用命令行（最简单）

**步骤 1: 初始化 Sweep**

在 Jupyter notebook cell 中运行：

```python
# 使用简化配置（推荐首次使用）
!wandb sweep configs/sweep_config_simple.yaml

# 或使用完整配置
# !wandb sweep configs/sweep_config.yaml
```

这会输出一个 sweep ID，例如：`username/project/sweep_id`

**步骤 2: 运行 Sweep Agent**

```python
# 替换为你的 sweep ID
SWEEP_ID = "username/project/sweep_id"
!wandb agent {SWEEP_ID}
```

#### 方法 2: 使用 Python API（更好的控制）

```python
import wandb

# 初始化 sweep
sweep_id = wandb.sweep(
    sweep="configs/sweep_config_simple.yaml",
    project="medical-tabular-data"  # 替换为你的项目名
)

# 运行 agent
wandb.agent(
    sweep_id=sweep_id,
    function=None,  # 使用配置文件中的 program
    count=10  # 运行 10 个实验，设置为 None 则无限运行
)
```

**查看示例 notebook**: 打开 `run_sweep.ipynb` 查看完整示例

### 在终端中运行

#### 1. 初始化 Sweep

使用完整配置：
```bash
wandb sweep configs/sweep_config.yaml
```

或使用简化配置（推荐首次使用）：
```bash
wandb sweep configs/sweep_config_simple.yaml
```

这会输出一个 sweep ID，例如：`username/project/sweep_id`

#### 2. 运行 Sweep Agent

```bash
wandb agent username/project/sweep_id
```

可以运行多个 agent 来并行执行多个实验：

```bash
# 终端 1
wandb agent username/project/sweep_id

# 终端 2
wandb agent username/project/sweep_id

# 终端 3
wandb agent username/project/sweep_id
```

#### 3. 在本地运行（不使用 wandb cloud）

```bash
wandb sweep --local configs/sweep_config.yaml
wandb agent <local-sweep-id>
```

## 自定义超参数搜索空间

编辑 `configs/sweep_config.yaml` 可以修改搜索空间。例如：

### 使用离散值而不是分布

```yaml
learning_rate:
  values: [0.0001, 0.0005, 0.001, 0.005, 0.01]
```

### 使用网格搜索

```yaml
method: grid  # 改为 grid 搜索
```

### 限制搜索的超参数

如果只想搜索部分超参数，可以在配置文件中注释掉不需要的参数。

## 查看结果

1. **在 wandb 网页界面查看**：
   - 访问 https://wandb.ai
   - 选择你的项目
   - 查看 sweep 页面，可以看到所有实验的结果对比

2. **本地结果**：
   - 每个实验的结果保存在 `output/` 目录下
   - 每个实验都有独立的目录，包含 checkpoints、plots、losses 等

## 优化指标

当前配置优化的是 `best_val_loss`（最佳验证损失，越小越好）。

如果你想优化其他指标（如 R²），可以修改配置文件：

```yaml
metric:
  name: best_val_r2/mlp  # 优化 MLP 模型的 R²
  goal: maximize  # R² 越大越好
```

## 注意事项

1. **搜索空间大小**：如果超参数组合太多，建议先使用 `sweep_config_simple.yaml` 进行初步实验
2. **计算资源**：每个实验都会训练所有模型，需要较长时间
3. **早停策略**：可以考虑在配置文件中添加早停配置
4. **固定参数**：某些参数（如 `seed`）可以通过命令行参数固定，不参与扫描

## 示例：只扫描学习率和批次大小

编辑 `configs/sweep_config_simple.yaml` 或创建新配置文件：

```yaml
program: sweep_main.py
method: grid
metric:
  name: best_val_loss
  goal: minimize
parameters:
  learning_rate:
    values: [0.0001, 0.001, 0.01]
  batch_size:
    values: [64, 128, 256]
  # 其他参数使用默认值或固定值
  epochs:
    value: 50
  mlp_hidden_dim_1:
    value: 256
  # ... 其他参数设为固定值
```

然后运行：

```bash
wandb sweep configs/sweep_config_simple.yaml
wandb agent <sweep-id>
```

## 故障排除

### 常见问题

1. **wandb 未登录**：运行 `wandb login`

2. **找不到配置文件**：确保在项目根目录运行命令，或使用完整路径

3. **超参数未应用**：检查 `sweep_main.py` 中的配置应用逻辑

4. **内存不足**：减少 `batch_size` 或并行运行的 agent 数量

5. **网络超时错误 (ReadTimeout)**：
   - 增加超时时间：`export WANDB_NETWORK_TIMEOUT=120`
   - 或使用离线模式：`export WANDB_MODE=offline`

6. **API 端点错误**（如 `Error: Invalid sweep config: Post "http://anaconda2.default.svc.cluster.local/search": EOF`）：
   
   这个问题通常发生在 Kubernetes 集群或网络代理环境中，wandb 尝试连接到错误的 API 端点。
   
   **解决方案 1：设置正确的 API 端点**
   ```bash
   export WANDB_BASE_URL=https://api.wandb.ai
   wandb sweep configs/sweep_config.yaml
   ```
   
   **解决方案 2：使用离线模式创建 sweep**
   ```bash
   export WANDB_MODE=offline
   wandb sweep configs/sweep_config.yaml
   ```
   注意：离线模式下，sweep 配置会保存在本地，稍后可以同步到 wandb 服务器。
   
   **解决方案 3：检查 wandb 配置文件**
   ```bash
   # 检查是否有错误的配置
   cat ~/.netrc  # 检查网络配置
   cat ~/.config/wandb/settings  # 检查 wandb 设置
   ```
   
   如果发现错误的 base_url 配置，可以删除或修改：
   ```bash
   # 删除 wandb 配置目录（会重置所有设置）
   rm -rf ~/.config/wandb
   wandb login  # 重新登录
   ```

7. **配置文件语法错误**：
   - 确保 YAML 格式正确
   - 对于 grid search，使用 `values:` 而不是 `value:`
   - 检查缩进是否正确（使用空格，不要使用 tab）
