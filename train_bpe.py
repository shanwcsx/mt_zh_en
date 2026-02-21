#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练BPE模型并应用到分词后的数据
"""

import os
from subword_nmt import learn_bpe
from subword_nmt import apply_bpe

def train_bpe():
    """训练BPE模型"""
    print("开始训练BPE模型...")
    
    # 训练BPE模型
    with open('wmt16/train_l', 'r', encoding='utf-8') as f_in, \
         open('wmt16/bpe.20000', 'w', encoding='utf-8') as f_out, \
         open('wmt16/vocab', 'w', encoding='utf-8') as f_vocab:
        learn_bpe.learn_bpe(f_in, f_out, 20000, is_dict=False)
    
    print("BPE模型训练完成！")

def apply_bpe_to_files():
    """应用BPE模型到分词后的数据"""
    print("开始应用BPE模型...")
    
    # 加载BPE模型
    with open('wmt16/bpe.20000', 'r', encoding='utf-8') as f_bpe:
        bpe = apply_bpe.BPE(f_bpe)
    
    # 处理训练集
    process_file(bpe, 'wmt16_cut/train_src.cut.txt', 'wmt16/train_src.bpe')
    process_file(bpe, 'wmt16_cut/train_trg.cut.txt', 'wmt16/train_trg.bpe')
    
    # 处理验证集
    process_file(bpe, 'wmt16_cut/val_src.cut.txt', 'wmt16/val_src.bpe')
    process_file(bpe, 'wmt16_cut/val_trg.cut.txt', 'wmt16/val_trg.bpe')
    
    # 处理测试集
    process_file(bpe, 'wmt16_cut/test_src.cut.txt', 'wmt16/test_src.bpe')
    process_file(bpe, 'wmt16_cut/test_trg.cut.txt', 'wmt16/test_trg.bpe')
    
    print("BPE模型应用完成！")

def process_file(bpe, input_file, output_file):
    """处理单个文件"""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if line:
                processed_line = bpe.process_line(line)
                f_out.write(processed_line + '\n')
    print(f"处理完成: {output_file}")

if __name__ == '__main__':
    train_bpe()
    apply_bpe_to_files()
