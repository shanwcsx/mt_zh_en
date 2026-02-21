#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成词汇表脚本
"""

import os
from collections import Counter

def load_data(data_dir, mode, max_samples=300000):
    """加载预处理后的数据"""
    src_file = os.path.join(data_dir, f'{mode}_src.bpe')
    trg_file = os.path.join(data_dir, f'{mode}_trg.bpe')
    
    src_lines = []
    trg_lines = []
    
    print(f"读取 {src_file}...")
    with open(src_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            if line.strip():
                src_lines.append(line.strip().split())
    
    print(f"读取 {trg_file}...")
    with open(trg_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            if line.strip():
                trg_lines.append(line.strip().split())
    
    return src_lines, trg_lines

def generate_vocab(src_lines, trg_lines, min_freq=2):
    """从数据中生成词汇表"""
    # 统计词频
    word_counts = Counter()
    
    # 统计源语言和目标语言的词频
    for line in src_lines:
        word_counts.update(line)
    for line in trg_lines:
        word_counts.update(line)
    
    # 生成词汇表，只包含频率大于等于min_freq的词
    vocab = {}
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

def main():
    """主函数"""
    # 配置路径
    data_dir = 'wmt16'
    vocab_file = os.path.join(data_dir, 'vocab')
    
    # 加载训练数据
    print("加载训练数据...")
    train_src, train_trg = load_data(data_dir, 'train', max_samples=None)
    print(f"加载了 {len(train_src)} 句训练数据")
    
    # 生成词汇表
    print("生成词汇表...")
    vocab = generate_vocab(train_src, train_trg)
    print(f"生成的词汇表大小: {len(vocab)}")
    
    # 保存词汇表
    os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word + '\n')
    print(f"词汇表已保存到: {vocab_file}")

if __name__ == "__main__":
    main()
