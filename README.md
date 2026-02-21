# Transformer中英翻译模型

## 项目简介

本项目实现了一个基于Transformer架构的中英翻译模型，使用深度学习技术实现高质量的神经机器翻译。模型采用经典的Transformer编码器-解码器架构，结合多头注意力机制，能够有效处理长距离依赖关系，实现流畅的中英互译。

## 技术栈

- **深度学习框架**: PyTorch
- **自然语言处理**: Sacremoses, NLTK
- **数据处理**: NumPy, Pandas
- **可视化**: Matplotlib
- **文本处理**: Subword-NMT (BPE)

## 项目特性

- ✅ 完整的Transformer架构实现
- ✅ 自定义位置编码和多头注意力机制
- ✅ BPE子词分割处理罕见词
- ✅ 中英文数据预处理流水线
- ✅ 灵活的模型配置和训练参数
- ✅ 模型保存与加载功能

## 项目结构

```
mt_zh_en/
├── wmt16/                    # 原始数据集目录
│   ├── train.zh              # 训练集中文
│   ├── train.en              # 训练集英文
│   ├── val.zh                # 验证集中文
│   ├── val.en                # 验证集英文
│   ├── test.zh               # 测试集中文
│   ├── test.en               # 测试集英文
│   ├── *.bpe                 # BPE处理后的数据
│   ├── bpe.20000             # BPE模型文件
│   └── vocab                 # 词汇表文件
├── wmt16_cut/                # 分词后数据目录
│   ├── train_src.cut.txt     # 训练源语言分词
│   ├── train_trg.cut.txt     # 训练目标语言分词
│   ├── val_src.cut.txt       # 验证源语言分词
│   ├── val_trg.cut.txt       # 验证目标语言分词
│   └── ...
├── data_multi30k.py          # 数据预处理脚本
├── train_bpe.py              # BPE模型训练脚本
├── train_transformer.py      # 主训练脚本
├── test_train.py             # 快速测试脚本
├── transformer_zh_en.ipynb   # Jupyter Notebook演示
├── merge_files.py            # 文件合并工具
└── README.md                 # 项目说明文档
```

## 模型架构

### Transformer核心组件

1. **位置编码 (Positional Encoding)**
   - 使用正弦和余弦函数编码绝对位置信息
   - 支持相对位置理解

2. **多头自注意力 (Multi-Head Attention)**
   - 并行计算多个注意力头
   - 捕获不同子空间的依赖关系
   - 标准注意力公式: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`

3. **前馈网络 (Feed Forward Network)**
   - 两层全连接网络
   - ReLU激活函数

4. **编码器-解码器架构**
   - 6层编码器堆叠
   - 6层解码器堆叠
   - 残差连接和层归一化

### 模型参数

| 参数 | 数值 | 说明 |
|------|------|------|
| 词嵌入维度 | 512 | 词向量维度 |
| 注意力头数 | 8 | 多头注意力头数 |
| 编码器层数 | 6 | 编码器堆叠层数 |
| 解码器层数 | 6 | 解码器堆叠层数 |
| 前馈网络维度 | 2048 | FFN隐藏层维度 |
| Dropout率 | 0.1 | 防止过拟合 |

## 数据预处理

### 1. 文本清洗与分词
- 中文字符级分词（支持中文分词工具）
- 英文Moses分词器处理
- 标点符号标准化

### 2. BPE子词分割
- 训练20000个BPE操作
- 平衡词汇表大小与稀有词处理
- 提高OOV（Out-of-Vocabulary）处理能力

### 3. 序列处理
- 特殊标记: `<sos>`, `<eos>`, `<pad>`, `<unk>`
- 序列长度统一至50
- 动态填充与截断

## 训练配置

### 优化器设置
- 优化器: Adam
- 学习率: 1e-4
- β₁: 0.9, β₂: 0.98
- ε: 1e-9

### 训练参数
- 批次大小: 32
- 训练轮数: 50
- 损失函数: 交叉熵损失（忽略padding）
- 学习率调度: StepLR (每10轮衰减0.5)

## 快速开始

### 环境准备

```bash
# 推荐使用conda环境
conda create -n transformer python=3.8
conda activate transformer

# 安装PyTorch (CUDA版本)
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia

# 安装其他依赖
pip install numpy matplotlib scikit-learn tqdm subword-nmt sacremoses nltk pandas
```

### 数据预处理

```bash
# 1. 执行数据预处理
python data_multi30k.py --pair_dir wmt16 --dest_dir wmt16_cut --src_lang zh --trg_lang en

# 2. 合并训练数据
python merge_files.py

# 3. 训练BPE模型
python train_bpe.py
```

### 模型训练

```bash
# 完整训练（推荐）
python train_transformer.py

# 快速测试（小模型）
python test_train.py
```

### 使用Jupyter Notebook

```bash
jupyter notebook transformer_zh_en.ipynb
```

## Flask Web应用

本项目还提供了一个基于Flask的Web界面，允许用户通过网页进行中英翻译。

### 功能特性
- 🌐 直观的网页界面
- ⚡ 实时翻译功能
- 📊 响应式设计
- 🔧 RESTful API接口

### 启动Web服务

```bash
# 安装Flask
pip install flask

# 启动Web服务
python app.py
```

服务启动后，访问 `http://127.0.0.1:5000` 即可使用网页界面进行翻译。

### API接口

- `GET /` - 主页
- `POST /translate` - 翻译接口
- `GET /health` - 健康检查

### API示例

```json
{
  "text": "我爱中国"
}
```

返回结果：
```json
{
  "success": true,
  "source": "我爱中国",
  "translation": "i love china"
}
```

## 扩展方向

### 模型优化
- 实现学习率预热策略
- 添加梯度裁剪机制
- 使用标签平滑技术

### 数据增强
- 集成更大规模的中英平行语料
- 使用回译技术扩充数据
- 数据清洗与质量过滤

### 架构升级
- 尝试更先进的Transformer变体（如T5, BART）
- 集成预训练模型（如BERT, RoBERTa）
- 实现知识蒸馏压缩模型

## 性能表现

在当前示例数据集上的初步结果：
- 训练准确率: ~64%
- 验证准确率: ~66%

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 致谢

- 基于Vaswani等人2017年论文《Attention is All You Need》
- 使用了开源的机器翻译数据集
- 感谢PyTorch社区提供的优秀框架
