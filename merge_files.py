#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并分词后的文件，确保编码正确
"""

import os

def merge_files():
    """合并中文和英文分词文件"""
    # 读取中文分词文件
    with open('wmt16_cut/train_src.cut.txt', 'r', encoding='utf-8') as f:
        zh_lines = f.readlines()
    
    # 读取英文分词文件
    with open('wmt16_cut/train_trg.cut.txt', 'r', encoding='utf-8') as f:
        en_lines = f.readlines()
    
    # 合并到train_l文件
    with open('wmt16/train_l', 'w', encoding='utf-8') as f:
        f.writelines(zh_lines)
        f.writelines(en_lines)
    
    print(f"成功合并文件: {len(zh_lines)} 中文行 + {len(en_lines)} 英文行")

if __name__ == '__main__':
    merge_files()
