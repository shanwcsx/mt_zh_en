#!/usr/bin/env python3
"""
中译英翻译Flask Web应用
提供网页界面进行中文到英文的翻译
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify

# 检查CUDA是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义特殊标记
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

# ============ 模型定义 ============

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        # 使用Embedding层实现位置编码
        self.pos_embedding = nn.Embedding(
            self.max_len,
            self.d_model,
            _weight=self.get_positional_encoding(self.max_len, self.d_model)
        )
        self.pos_embedding.weight.requires_grad_(False)  # 不更新位置编码的权重

    def get_positional_encoding(self, max_length, hidden_size):
        """计算位置编码"""
        pe = torch.zeros(max_length, hidden_size)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2)
            * -(torch.log(torch.Tensor([10000.0])) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        """前向传播"""
        batch_size, seq_len, _ = x.size()
        
        # 生成位置 IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        
        # 获取位置编码
        pos_embeds = self.pos_embedding(position_ids)
        
        # 与输入相加
        x = x + pos_embeds
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.d_k = d_model // nhead
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q_seq_len = q.size(1)
        k_seq_len = k.size(1)
        
        # 线性变换并分多头
        q = self.q_linear(q).reshape(batch_size, q_seq_len, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_linear(k).reshape(batch_size, k_seq_len, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_linear(v).reshape(batch_size, k_seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # 确保掩码形状正确
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            mask = mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(mask == 0, -1e4)  # 使用在半精度范围内的值
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 应用注意力
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().reshape(batch_size, q_seq_len, self.d_model)
        output = self.out_linear(output)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, d_model, nhead, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, src_mask):
        # 自注意力
        attn_output, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, d_model, nhead, d_ff):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, enc_output, tgt_mask, src_tgt_mask):
        # 自注意力
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_tgt_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Encoder(nn.Module):
    """编码器"""
    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, src_mask):
        x = self.embedding(src)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x


class Decoder(nn.Module):
    """解码器"""
    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, d_ff) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)

    def forward(self, tgt, enc_output, tgt_mask, src_tgt_mask):
        x = self.embedding(tgt)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_tgt_mask)
        
        return x


class Transformer(nn.Module):
    """Transformer模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_layers, d_ff):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, nhead, num_layers, d_ff)
        self.decoder = Decoder(tgt_vocab_size, d_model, nhead, num_layers, d_ff)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask, src_tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_tgt_mask)
        output = self.generator(dec_output)
        return output


# ============ 工具函数 ============

def load_vocab(vocab_file):
    """加载词汇表"""
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            word = line.strip()
            if word:
                vocab[word] = idx
    
    # 添加特殊标记，与训练时保持一致
    special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    
    return vocab


def load_model(model_path, vocab, device):
    """加载训练好的模型"""
    # 模型参数
    d_model = 512
    nhead = 8
    num_layers = 6
    d_ff = 2048
    src_vocab_size = len(vocab)
    tgt_vocab_size = len(vocab)
    
    # 创建模型
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, nhead, num_layers, d_ff)
    model = model.to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型已加载，epoch: {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}, accuracy: {checkpoint['accuracy']:.4f}")
    return model


def translate(model, src_seq, vocab, reverse_vocab, device, max_len=30):
    """将中文翻译成英文（优化版）"""
    model.eval()
    
    # 预处理输入
    src_tokens = [vocab.get(word, vocab[UNK_TOKEN]) for word in src_seq]
    src_tokens = [vocab[SOS_TOKEN]] + src_tokens + [vocab[EOS_TOKEN]]
    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
    
    # 生成源语言掩码
    src_mask = (src_tensor != vocab[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
    
    # 只计算一次编码器输出，重复使用
    enc_output = model.encoder(src_tensor, src_mask)
    
    # 源语言-目标语言掩码（只计算一次）
    src_tgt_mask = (src_tensor != vocab[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
    
    # 初始化目标序列
    tgt_seq = [vocab[SOS_TOKEN]]
    
    # 禁用梯度计算以提高速度
    with torch.no_grad():
        for i in range(max_len):
            tgt_tensor = torch.tensor([tgt_seq], dtype=torch.long).to(device)
            
            # 生成目标语言掩码
            seq_len = tgt_tensor.size(1)
            tgt_mask = (tgt_tensor != vocab[PAD_TOKEN]).unsqueeze(1).unsqueeze(3)
            tgt_sub_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
            tgt_mask = tgt_mask & tgt_sub_mask
            
            # 直接调用解码器和生成器，避免完整模型前向传播的开销
            dec_output = model.decoder(tgt_tensor, enc_output, tgt_mask, src_tgt_mask)
            output = model.generator(dec_output)
            
            # 预测下一个词
            next_token = output[0, -1, :].argmax().item()
            tgt_seq.append(next_token)
            
            # 如果遇到EOS标记，停止生成
            if next_token == vocab[EOS_TOKEN]:
                break
    
    # 转换为文本
    translation = [reverse_vocab[token] for token in tgt_seq if token not in [vocab[SOS_TOKEN], vocab[EOS_TOKEN], vocab[PAD_TOKEN]]]
    return ' '.join(translation)


# ============ Flask应用 ============

app = Flask(__name__)

# 全局变量存储模型和词汇表
model = None
vocab = None
reverse_vocab = None


def initialize_model():
    """初始化模型和词汇表"""
    global model, vocab, reverse_vocab
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 配置路径（使用绝对路径）
    data_dir = os.path.join(script_dir, 'wmt16')
    vocab_file = os.path.join(data_dir, 'vocab')
    model_path = os.path.join(script_dir, 'checkpoints', 'best_model.pt')
    
    # 检查文件是否存在
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"词汇表文件不存在: {vocab_file}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载词汇表
    print("加载词汇表...")
    vocab = load_vocab(vocab_file)
    reverse_vocab = {v: k for k, v in vocab.items()}
    print(f"词汇表大小: {len(vocab)}")
    
    # 加载模型
    print("加载模型...")
    model = load_model(model_path, vocab, device)
    print("模型初始化完成！")


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate_text():
    """翻译接口"""
    try:
        # 获取输入文本
        data = request.get_json()
        text = data.get('text', '').strip()
        
        # 检查输入是否为空
        if not text:
            return jsonify({'error': '输入不能为空'}), 400
        
        # 预处理输入
        src_seq = list(text.strip())
        
        # 翻译
        translation = translate(model, src_seq, vocab, reverse_vocab, device)
        
        # 返回结果
        return jsonify({
            'success': True,
            'source': text,
            'translation': translation
        })
    
    except Exception as e:
        print(f"翻译错误: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """健康检查接口"""
    return jsonify({'status': 'ok', 'device': str(device)})


if __name__ == '__main__':
    # 初始化模型
    print("="*60)
    print("中译英翻译Flask Web应用")
    print("="*60)
    
    try:
        initialize_model()
        print("\n" + "="*60)
        print("启动Flask服务器...")
        print("="*60)
        print("访问地址: http://127.0.0.1:5000")
        print("="*60 + "\n")
        
        # 启动Flask应用
        app.run(host='0.0.0.0', port=5000, debug=False)
    
    except Exception as e:
        print(f"初始化失败: {e}")
        exit(1)
