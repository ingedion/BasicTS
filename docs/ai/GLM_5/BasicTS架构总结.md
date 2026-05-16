# BasicTS 代码架构详细总结

## 一、架构概览

BasicTS 是一个公平且可扩展的时间序列分析基准库和工具包,支持时空预测、长期时间序列预测、分类和填充等多种任务。其核心设计理念是通过三层架构实现高度解耦和可扩展性。

### 核心特点

1. **统一标准化的流程**: 提供公平、全面的模型复现和比较平台
2. **用户友好的扩展接口**: 快速设计和评估新模型
3. **三层架构设计**: 执行器层、任务流层、回调层,实现功能解耦
4. **配置驱动**: 所有细节通过配置文件控制
5. **多设备支持**: CPU、GPU和分布式训练

---

## 二、目录结构与模块划分

```
src/basicts/
├── configs/           # 配置模块
├── data/              # 数据集模块
├── metrics/           # 评估指标模块
├── models/            # 模型实现
├── modules/           # 通用组件模块
├── runners/           # 执行器模块(核心)
├── scaler/            # 数据缩放器模块
├── utils/             # 工具函数
└── launcher.py        # 启动器
```

---

## 三、核心模块详解

### 3.1 配置模块 (configs/)

**位置**: `src/basicts/configs/`

**核心文件**:
- `base_config.py`: 基础配置类 `BasicTSConfig`
- `model_config.py`: 模型配置基类 `BasicTSModelConfig`
- `tsf_config.py`: 时间序列预测配置 `BasicTSForecastingConfig`
- `tsc_config.py`: 分类配置 `BasicTSClassificationConfig`
- `tsi_config.py`: 填充配置 `BasicTSImputationConfig`
- `tsfm_config.py`: 基础模型配置 `BasicTSFoundationModelConfig`

**设计特点**:

1. **基于 dataclass 和 EasyDict**: 结合类型提示和动态属性访问
2. **配置序列化**: 支持保存为 JSON 格式,使用 MD5 标识唯一配置
3. **参数打包机制**: 自动将配置参数打包到对应组件

```python
@dataclass(init=False)
class BasicTSConfig(EasyDict):
    # 必需字段
    model: type                        # 模型类
    model_config: BasicTSModelConfig  # 模型配置
    dataset_name: str                  # 数据集名称
    
    # 通用配置
    gpus: Optional[str]               # GPU设备
    seed: int                         # 随机种子
    
    # 数据集配置
    dataset_type: type                # 数据集类型
    dataset_params: dict              # 数据集参数
    
    # 训练配置
    num_epochs: int                   # 训练轮数
    loss: Union[str, Callable]        # 损失函数
    optimizer: type                   # 优化器
    
    # 回调和任务流
    taskflow: BasicTSTaskFlow         # 任务流
    callbacks: List[BasicTSCallback]  # 回调列表
```

**参数分类**:
- **训练独立参数**: 不影响训练结果的参数(如 batch_size、num_workers)
- **训练依赖参数**: 影响模型训练的参数(如学习率、模型结构)

---

### 3.2 数据集模块 (data/)

**位置**: `src/basicts/data/`

**核心类**:

1. **BasicTSDataset** (`base_dataset.py`)
   - 所有数据集的基类
   - 定义数据集的基本接口

2. **BasicTSForecastingDataset** (`tsf_dataset.py`)
   - 时间序列预测数据集
   - 处理输入序列和目标序列的切片

3. **UEADataset** (`uea_dataset.py`)
   - UEA 时间序列分类数据集

4. **BasicTSImputationDataset** (`tsi_dataset.py`)
   - 时间序列填充数据集

5. **BLAST** (`blast.py`)
   - 大规模语料数据集

**数据集设计**:

```python
class BasicTSForecastingDataset(BasicTSDataset):
    def __init__(self, 
                 dataset_name: str,
                 input_len: int,       # 输入序列长度
                 output_len: int,      # 输出序列长度
                 mode: BasicTSMode,    # train/val/test
                 use_timestamps: bool, # 是否使用时间戳
                 memmap: bool):        # 是否使用内存映射
    
    def __getitem__(self, index: int) -> dict:
        # 返回字典: {'inputs': ..., 'targets': ...}
        # 可选: 'inputs_timestamps', 'targets_timestamps'
```

**数据加载策略**:
- 支持 `.npy` 格式的预处理数据
- 可选内存映射模式(memmap)处理大数据集
- 自动划分训练/验证/测试集

---

### 3.3 数据缩放器模块 (scaler/)

**位置**: `src/basicts/scaler/`

**核心类**:

1. **BasicTSScaler** (`base_scaler.py`)
   - 缩放器基类,定义标准化接口

2. **ZScoreScaler** (`z_score_scaler.py`)
   - Z-score 标准化

3. **MinMaxScaler** (`min_max_scaler.py`)
   - Min-Max 归一化

**缩放器接口**:

```python
@dataclass
class BasicTSScaler:
    norm_each_channel: bool  # 是否每个通道独立归一化
    rescale: bool            # 是否重新缩放
    stats: dict              # 存储统计信息
    
    def fit(self, data: np.ndarray) -> None:
        """在训练数据上拟合缩放器"""
        
    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        """应用缩放变换"""
        
    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        """逆变换,恢复原始尺度"""
```

**使用场景**:
- 训练前对数据进行标准化
- 推理后对预测结果进行反归一化
- 在任务流(taskflow)的预处理和后处理阶段调用

---

### 3.4 评估指标模块 (metrics/)

**位置**: `src/basicts/metrics/`

**支持的指标**:

```python
ALL_METRICS = {
    'MAE': masked_mae,      # 平均绝对误差
    'MSE': masked_mse,      # 均方误差
    'RMSE': masked_rmse,    # 均方根误差
    'MAPE': masked_mape,    # 平均绝对百分比误差
    'WAPE': masked_wape,    # 加权平均绝对百分比误差
    'SMAPE': masked_smape,  # 对称平均绝对百分比误差
    'R2': masked_r2,        # R平方
    'CORR': masked_corr,    # 相关系数
    'HUBER': masked_huber,  # Huber损失
    'Accuracy': accuracy,   # 分类准确率
}
```

**特点**:
- 所有指标支持掩码(mask)处理,忽略缺失值
- 支持自定义指标函数
- 计量器(Meter)系统跟踪指标值

---

### 3.5 模型模块 (models/)

**位置**: `src/basicts/models/`

**模型组织结构**:
```
models/
├── STID/              # 模型名称
│   ├── __init__.py    # 导出模型类和配置类
│   ├── arch/          # 模型架构实现
│   │   └── stid_arch.py
│   └── config/        # 模型配置
│       └── stid_config.py
├── TimesNet/
├── iTransformer/
└── ... (50+ 模型)
```

**模型开发规范**:

1. **模型类**:
```python
class STID(nn.Module):
    def __init__(self, config: STIDConfig):
        """初始化模型,使用配置对象"""
        
    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播
        Args:
            inputs: 输入张量 [batch, input_len, num_features]
        Returns:
            prediction: 预测结果 [batch, output_len, num_features]
        """
```

2. **配置类**:
```python
@dataclass
class STIDConfig(BasicTSModelConfig):
    input_len: int
    output_len: int
    num_features: int
    # ... 其他超参数
```

**支持的模型类型**:
- 时空预测模型: STGCN, DCRNN, GWNet, MTGNN, AGCRN 等
- 长期预测模型: TimesNet, iTransformer, PatchTST, Autoformer 等
- 通用预测模型: TimeMoE, ChronosBolt, MOIRAI
- 其他: LightGBM, CatBoost, DeepAR, N-BEATS 等

---

### 3.6 通用组件模块 (modules/)

**位置**: `src/basicts/modules/`

**子模块**:

1. **embed/**: 嵌入层
   - `tst_embed.py`: 时间序列嵌入

2. **norm/**: 归一化层
   - `layer_norm.py`: 层归一化
   - `revin.py`: 可逆实例归一化
   - `rmsnorm.py`: RMS 归一化
   - `stnorm.py`: 时空归一化

3. **transformer/**: Transformer 组件
   - `attentions/`: 注意力机制
     - `multi_head_attention.py`: 多头注意力
     - `auto_correlation.py`: 自相关注意力
     - `prob_attention.py`: 概率注意力
   - `encoder.py`: 编码器
   - `decoder.py`: 解码器
   - `rope.py`: 旋转位置编码
   - `kv_cache.py`: KV 缓存

4. **其他组件**:
   - `activations.py`: 激活函数
   - `decomposition.py`: 序列分解
   - `mlps.py`: MLP 层

**使用示例**:
```python
from basicts.modules import MLPLayer, ResMLPLayer
from basicts.modules.transformer import MultiHeadAttention
```

---

### 3.7 执行器模块 (runners/) - 核心

**位置**: `src/basicts/runners/`

这是 BasicTS 的**核心模块**,实现了训练和评估的完整流程。

#### 3.7.1 架构设计:三层架构

```
┌─────────────────────────────────────┐
│  执行器层 (BasicTSRunner)            │  ← 通用训练流程,不应修改
│  - 通用流程:前向、损失、反向、优化      │
└─────────────────────────────────────┘
           ↓ 调用
┌─────────────────────────────────────┐
│  任务流层 (BasicTSTaskFlow)          │  ← 任务相关步骤,较少修改
│  - preprocess: 数据预处理             │
│  - postprocess: 结果后处理            │
│  - get_weight: 损失权重               │
└─────────────────────────────────────┘
           ↓ 调用
┌─────────────────────────────────────┐
│  回调层 (BasicTSCallback)            │  ← 扩展功能,推荐自定义
│  - early_stopping: 早停              │
│  - clip_grad: 梯度裁剪                │
│  - curriculum_learning: 课程学习      │
│  - selective_learning: 选择性学习     │
└─────────────────────────────────────┘
```

#### 3.7.2 BasicTSRunner 核心流程

**初始化阶段** (`__init__`):
```python
def __init__(self, cfg: BasicTSConfig):
    # 1. 设置环境(随机种子、CUDA等)
    self.set_env(cfg)
    
    # 2. 创建模型
    self.model = Builder._build_model(cfg, self.logger)
    
    # 3. 创建优化器和学习率调度器
    self.optimizer = Builder._build_optimizer(cfg, self.model)
    self.lr_scheduler = Builder._build_lr_scheduler(cfg, self.optimizer)
    
    # 4. 创建数据缩放器
    self.scaler = Builder._build_scaler(cfg)
    
    # 5. 初始化回调处理器
    self.callback_handler = BasicTSCallbackHandler(cfg.callbacks)
    
    # 6. 初始化任务流
    self.taskflow = cfg.taskflow
```

**训练流程** (`train`):
```python
def train(self):
    # 1. 训练开始
    self.on_train_start()
    self.callback_handler.trigger("on_train_start", self)
    
    # 2. 训练循环
    self._train_loop()
    
    # 3. 训练结束
    self.callback_handler.trigger("on_train_end", self)
    self.on_train_end()
    
    # 4. 训练后评估
    if self.cfg.eval_after_train:
        self.eval(best_model_path)
```

**训练循环** (`_train_loop`):
```python
def _train_loop(self):
    while self.global_steps <= self.num_steps:
        # 1. Epoch 开始
        self.on_epoch_start(self.epoch)
        self.callback_handler.trigger("on_epoch_start", self)
        
        for data in self.train_data_loader:
            # 2. Step 开始
            self.on_step_start(self.global_steps)
            self.callback_handler.trigger("on_step_start", self)
            
            # 3. 数据预处理(任务流)
            data = self.taskflow.preprocess(self, data)
            
            # 4. 模型前向
            with self.amp_ctx:
                forward_return = self._forward(self.model, data)
            
            # 5. 计算损失前回调
            self.callback_handler.trigger("on_compute_loss", self)
            
            # 6. 计算损失
            loss = self._metric_forward(self.loss, forward_return)
            loss_weight = self.taskflow.get_weight(forward_return)
            
            # 7. 反向传播前回调
            self.callback_handler.trigger("on_backward", self)
            
            # 8. 反向传播
            loss.backward()
            
            # 9. 优化器更新前回调
            self.callback_handler.trigger("on_optimizer_step", self)
            
            # 10. 优化器更新
            self._optimizer_step()
            
            # 11. 后处理(任务流)
            forward_return = self.taskflow.postprocess(self, forward_return)
            
            # 12. 计算指标
            for metric_name, metric_fn in self.metrics.items():
                metric_value = self._metric_forward(metric_fn, forward_return)
            
            # 13. Step 结束
            self.callback_handler.trigger("on_step_end", self)
            self.on_step_end(self.global_steps)
        
        # 14. Epoch 结束
        self.callback_handler.trigger("on_epoch_end", self)
        self.on_epoch_end(self.epoch)
```

**评估流程** (`_eval_loop`):
```python
def _eval_loop(self, mode: BasicTSMode):
    for step, data in enumerate(data_loader):
        # 1. 预处理
        data = self.taskflow.preprocess(self, data)
        
        # 2. 前向传播
        forward_return = self._forward(self.model, data)
        
        # 3. 计算损失
        loss = self._metric_forward(self.loss, forward_return)
        
        # 4. 后处理
        forward_return = self.taskflow.postprocess(self, forward_return)
        
        # 5. 保存结果
        if mode == BasicTSMode.EVAL:
            self._save_results(step, forward_return)
        
        # 6. 计算指标
        for metric_name, metric_func in self.metrics.items():
            metric_value = self._metric_forward(metric_func, forward_return)
```

#### 3.7.3 Builder 类

**位置**: `src/basicts/runners/builder.py`

**职责**: 构建各种组件

```python
class Builder:
    @staticmethod
    def _build_model(cfg, logger) -> nn.Module:
        """构建模型,支持 DDP 和 torch.compile"""
        
    @staticmethod
    def _build_data_loader(cfg, mode, logger) -> DataLoader:
        """构建数据加载器,支持分布式采样"""
        
    @staticmethod
    def _build_dataset(cfg, mode) -> Dataset:
        """构建数据集"""
        
    @staticmethod
    def _build_optimizer(cfg, model) -> Optimizer:
        """构建优化器"""
        
    @staticmethod
    def _build_lr_scheduler(cfg, optimizer) -> LRScheduler:
        """构建学习率调度器"""
        
    @staticmethod
    def _build_scaler(cfg) -> BasicTSScaler:
        """构建数据缩放器"""
```

---

### 3.8 任务流模块 (runners/taskflow/)

**位置**: `src/basicts/runners/taskflow/`

#### 3.8.1 BasicTSTaskFlow 基类

```python
class BasicTSTaskFlow(ABC):
    @abstractmethod
    def preprocess(self, runner, data: Dict) -> Dict:
        """数据预处理:归一化、缺失值处理等"""
        
    @abstractmethod
    def postprocess(self, runner, forward_return: Dict) -> Dict:
        """结果后处理:反归一化、argmax等"""
        
    @abstractmethod
    def get_weight(self, forward_return: Dict) -> float:
        """获取损失权重:用于正确计算平均损失"""
```

#### 3.8.2 BasicTSForecastingTaskFlow 预测任务流

```python
class BasicTSForecastingTaskFlow(BasicTSTaskFlow):
    def preprocess(self, runner, data):
        # 1. 创建缺失值掩码
        inputs_mask = null_val_mask(data['inputs'], runner.cfg.null_val)
        targets_mask = null_val_mask(data['targets'], runner.cfg.null_val)
        
        # 2. 数据归一化
        if runner.scaler is not None:
            data['inputs'] = runner.scaler.transform(data['inputs'])
            data['targets'] = runner.scaler.transform(data['targets'])
        
        # 3. 填充缺失值
        data['inputs'] = torch.where(inputs_mask, data['inputs'], 0.0)
        data['targets'] = torch.where(targets_mask, data['targets'], 0.0)
        
        data['targets_mask'] = targets_mask
        return data
    
    def postprocess(self, runner, forward_return):
        # 反归一化
        if runner.cfg.rescale and runner.scaler is not None:
            forward_return['prediction'] = runner.scaler.inverse_transform(
                forward_return['prediction'])
            forward_return['targets'] = runner.scaler.inverse_transform(
                forward_return['targets'])
        return forward_return
    
    def get_weight(self, forward_return):
        # 返回有效样本数量
        return forward_return['targets_mask'].sum().item()
```

#### 3.8.3 其他任务流

- **BasicTSClassificationTaskFlow**: 分类任务流
- **BasicTSImputationTaskFlow**: 填充任务流

---

### 3.9 回调模块 (runners/callback/)

**位置**: `src/basicts/runners/callback/`

#### 3.9.1 BasicTSCallback 基类

```python
class BasicTSCallback:
    # 训练阶段
    def on_train_start(self, runner, *args, **kwargs): pass
    def on_train_end(self, runner, *args, **kwargs): pass
    
    # Epoch 阶段
    def on_epoch_start(self, runner, *args, **kwargs): pass
    def on_epoch_end(self, runner, *args, **kwargs): pass
    
    # Step 阶段
    def on_step_start(self, runner, *args, **kwargs): pass
    def on_step_end(self, runner, *args, **kwargs): pass
    
    # 验证阶段
    def on_validate_start(self, runner, *args, **kwargs): pass
    def on_validate_end(self, runner, *args, **kwargs): pass
    
    # 测试阶段
    def on_test_start(self, runner, *args, **kwargs): pass
    def on_test_end(self, runner, *args, **kwargs): pass
    
    # 训练细节
    def on_compute_loss(self, runner, *args, **kwargs): pass
    def on_backward(self, runner, *args, **kwargs): pass
    def on_optimizer_step(self, runner, *args, **kwargs): pass
```

#### 3.9.2 BasicTSCallbackHandler

```python
class BasicTSCallbackHandler:
    def __init__(self, callbacks: list):
        self.callbacks = callbacks
    
    def trigger(self, event_name: str, runner, *args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, event_name, None)
            if method is not None:
                method(runner, *args, **kwargs)
```

#### 3.9.3 内置回调

1. **EarlyStopping** (`early_stopping.py`)
   - 早停机制,防止过拟合

2. **GradientClipping** (`clip_grad.py`)
   - 梯度裁剪,防止梯度爆炸

3. **CurriculumLearning** (`curriculum_learning.py`)
   - 课程学习,逐步增加任务难度

4. **SelectiveLearning** (`selective_learning.py`)
   - 选择性学习(NeurIPS'25),缓解过拟合

5. **GradAccumulation** (`grad_accumulation.py`)
   - 梯度累积,支持大 batch 训练

6. **AddAuxiliaryLoss** (`add_aux_loss.py`)
   - 添加辅助损失

7. **NoBP** (`no_bp.py`)
   - 阻止某些参数的反向传播

---

### 3.10 启动器 (launcher.py)

**位置**: `src/basicts/launcher.py`

**核心类**: `BasicTSLauncher`

**功能**:

1. **启动训练** (`launch_training`):
```python
@staticmethod
def launch_training(cfg: BasicTSConfig, node_rank: int = 0):
    # 1. 保存配置
    if node_rank == 0:
        cfg.save()
    
    # 2. 设置设备
    if cfg.gpus:
        set_device_type("gpu")
        set_visible_devices(cfg.gpus)
    else:
        set_device_type("cpu")
    
    # 3. 启动分布式训练
    train_dist = dist_wrap(
        training_func,
        node_num=cfg.dist_node_num,
        device_num=cfg.gpu_num,
        node_rank=node_rank
    )
    train_dist(cfg)
```

2. **启动评估** (`launch_evaluation`):
```python
@staticmethod
def launch_evaluation(cfg, ckpt_path, gpus=None, batch_size=None):
    # 1. 设置设备和批次大小
    set_device_type("gpu" if gpus else "cpu")
    if gpus:
        set_visible_devices(gpus)
    if batch_size:
        cfg.test_batch_size = batch_size
    
    # 2. 创建执行器
    runner = BasicTSRunner(cfg)
    runner.init_logger(logger_name="BasicTS-evaluation")
    
    # 3. 执行评估
    runner.eval(ckpt_path)
```

---

## 四、工具模块 (utils/)

**位置**: `src/basicts/utils/`

**核心工具**:

1. **constants.py**: 常量定义
   - `BasicTSMode`: 训练模式枚举(TRAIN, VAL, TEST, EVAL)
   - `BasicTSTask`: 任务类型枚举
   - `RunnerStatus`: 执行器状态枚举

2. **meter_pool.py**: 指标池
   - 管理和更新训练指标
   - 支持 TensorBoard 可视化

3. **misc.py**: 杂项工具
   - `clock`: 计时装饰器
   - `check_nan_inf`: 检查 NaN 和 Inf
   - `partial_func`: 部分函数

4. **serialization.py**: 序列化工具
   - `load_adj`: 加载邻接矩阵
   - `load_pkl` / `dump_pkl`: pickle 操作
   - `get_regular_settings`: 获取数据集常规设置

5. **mask.py**: 掩码工具
   - `null_val_mask`: 创建缺失值掩码

---

## 五、工作流程示例

### 5.1 完整训练流程

```python
# 1. 定义配置
config = BasicTSForecastingConfig(
    model=STID,
    model_config=STIDConfig(
        input_len=336,
        output_len=336,
        num_features=862
    ),
    dataset_name="PEMS04",
    gpus="0",
    num_epochs=100,
    batch_size=64
)

# 2. 启动训练
BasicTSLauncher.launch_training(config)
```

### 5.2 内部执行流程

```
BasicTSLauncher.launch_training()
    ↓
training_func()
    ↓
BasicTSRunner.__init__()          # 初始化组件
    ↓
runner.train()                     # 开始训练
    ↓
├─ on_train_start()               # 初始化训练
├─ _train_loop()                  # 训练循环
│   ├─ on_epoch_start()
│   ├─ for data in dataloader:
│   │   ├─ on_step_start()
│   │   ├─ taskflow.preprocess()  # 数据预处理
│   │   ├─ _forward()             # 模型前向
│   │   ├─ on_compute_loss()
│   │   ├─ loss computation        # 计算损失
│   │   ├─ on_backward()
│   │   ├─ loss.backward()         # 反向传播
│   │   ├─ on_optimizer_step()
│   │   ├─ optimizer.step()        # 优化器更新
│   │   ├─ taskflow.postprocess() # 结果后处理
│   │   └─ on_step_end()
│   └─ on_epoch_end()
└─ on_train_end()
```

---

## 六、设计模式与优势

### 6.1 核心设计模式

1. **策略模式**: 任务流(Taskflow)定义不同的处理策略
2. **观察者模式**: 回调(Callback)监听训练过程中的事件
3. **建造者模式**: Builder 负责构建各种组件
4. **模板方法模式**: Runner 定义训练骨架,子类可扩展

### 6.2 架构优势

1. **高度解耦**: 三层架构实现关注点分离
2. **易于扩展**: 
   - 新模型:实现模型类和配置类
   - 新任务:继承 BasicTSTaskFlow
   - 新功能:继承 BasicTSCallback
3. **公平比较**: 统一的流程确保不同模型的公平比较
4. **配置驱动**: 所有超参数通过配置管理,便于复现
5. **分布式支持**: 基于 EasyTorch 实现多机多卡训练

---

## 七、关键特性

### 7.1 混合精度训练

```python
# 自动检测并启用 AMP
self.use_amp = self.ptdtype in [torch.bfloat16, torch.float16]
self.amp_ctx = torch.amp.autocast(device_type="cuda", dtype=self.ptdtype)
self.amp_scaler = torch.amp.GradScaler(enabled=self.use_amp)
```

### 7.2 模型编译

```python
# 支持 torch.compile (PyTorch 2.0+)
if cfg.compile_model and version.parse(torch.__version__) >= version.parse("2.0"):
    model = torch.compile(model)
```

### 7.3 分布式训练

```python
# 自动包装 DDP
if torch.distributed.is_initialized():
    model = DDP(model, device_ids=[get_local_rank()])
```

### 7.4 梯度累积

通过 `GradAccumulation` 回调实现:
```python
# 小 batch 累积多次梯度再更新
if step % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### 7.5 检查点管理

- 自动保存最佳模型(基于验证集指标)
- 支持断点续训
- 灵活的保存策略(按 epoch 或步数)

---

## 八、扩展开发指南

### 8.1 添加新模型

1. 创建模型目录: `src/basicts/models/YourModel/`
2. 实现模型架构: `arch/your_model_arch.py`
3. 定义配置类: `config/your_model_config.py`
4. 导出组件: `__init__.py`

```python
# __init__.py
from .arch.your_model_arch import YourModel
from .config.your_model_config import YourModelConfig
```

### 8.2 添加新任务

1. 继承 `BasicTSTaskFlow`
2. 实现 `preprocess`、`postprocess`、`get_weight`
3. 创建对应的数据集类
4. 创建配置类

### 8.3 添加新回调

1. 继承 `BasicTSCallback`
2. 实现需要的钩子函数
3. 在配置中添加回调

```python
class MyCallback(BasicTSCallback):
    def on_epoch_end(self, runner, *args, **kwargs):
        # 自定义逻辑
        pass

config.callbacks = [MyCallback()]
```

---

## 九、总结

BasicTS 通过三层架构设计(执行器-任务流-回调)实现了高度的模块化和可扩展性。其核心优势在于:

1. **统一的基准平台**: 公平比较不同模型
2. **极简的开发体验**: 只需关注模型结构本身
3. **灵活的扩展机制**: 通过回调轻松添加新功能
4. **完善的工程实践**: 支持分布式、混合精度、梯度累积等

这种设计使得 BasicTS 既适合作为学术研究的基准测试工具,也适合工业界的实际应用开发。

---

**文档版本**: BasicTS v1.1.0  
**生成时间**: 2025年  
**作者**: AI Assistant
