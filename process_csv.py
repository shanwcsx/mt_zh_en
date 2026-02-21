import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 读取CSV文件（无表头）
csv_path = r'd:\shawn\computer\1.PythonAI2026项目班\每天上课代码及课堂笔记\day31\mt_zh_en\data\wmt_zh_en_training_corpus.csv'
df = pd.read_csv(csv_path, header=None)

print(f'总行数: {len(df)}')
print(f'列数: {len(df.columns)}')

# CSV格式：第一行是索引行，从第二行开始，第一列是中文，第二列是英文
# 跳过第一行（索引行）
df_data = df.iloc[1:, :].copy()

# 清洗数据：只保留中英文都完整的行
df_clean = df_data.dropna(subset=[0, 1])

print(f'\n清洗前:')
print(f'  中文句子数: {df_data[0].dropna().shape[0]}')
print(f'  英文句子数: {df_data[1].dropna().shape[0]}')

print(f'\n清洗后:')
print(f'  完整句子对数: {len(df_clean)}')

zh_texts = df_clean[0].tolist()
en_texts = df_clean[1].tolist()

# 划分数据集：训练集90%，验证集5%，测试集5%
# 先划分出训练集和临时集
train_zh, temp_zh, train_en, temp_en = train_test_split(
    zh_texts, en_texts, test_size=0.1, random_state=42
)

# 再将临时集划分为验证集和测试集
val_zh, test_zh, val_en, test_en = train_test_split(
    temp_zh, temp_en, test_size=0.5, random_state=42
)

print(f'\n数据集划分:')
print(f'  训练集: {len(train_zh)} 条')
print(f'  验证集: {len(val_zh)} 条')
print(f'  测试集: {len(test_zh)} 条')

# 输出目录
output_dir = r'd:\shawn\computer\1.PythonAI2026项目班\每天上课代码及课堂笔记\day31\mt_zh_en\wmt16'

# 定义数据集
datasets = [
    ('train', train_zh, train_en),
    ('val', val_zh, val_en),
    ('test', test_zh, test_en)
]

# 写入文件
for name, zh_data, en_data in datasets:
    # 写入中文文件
    zh_output_path = os.path.join(output_dir, f'{name}.zh')
    with open(zh_output_path, 'w', encoding='utf-8') as f:
        for zh_text in zh_data:
            f.write(str(zh_text) + '\n')
    
    # 写入英文文件
    en_output_path = os.path.join(output_dir, f'{name}.en')
    with open(en_output_path, 'w', encoding='utf-8') as f:
        for en_text in en_data:
            f.write(str(en_text) + '\n')

print(f'\n成功生成文件:')
for name, _, _ in datasets:
    print(f'  - {output_dir}\\{name}.zh')
    print(f'  - {output_dir}\\{name}.en')
