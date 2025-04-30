# File: scripts/gen_annotations_csv.py

import os
import csv

# 你的 Datasets 根目录
DATASETS_ROOT = 'Datasets'
# 输出 CSV 路径
CSV_PATH = os.path.join(DATASETS_ROOT, 'annotations.csv')

with open(CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # 写入表头
    writer.writerow(['style', 'image_filename'])

    # 遍历每个风格子目录
    for style in os.listdir(DATASETS_ROOT):
        style_dir = os.path.join(DATASETS_ROOT, style)
        if not os.path.isdir(style_dir):
            continue

        # 遍历该风格下的所有文件
        for fname in os.listdir(style_dir):
            # 过滤非图片文件（可根据需要修改扩展名列表）
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            # 写入一行：风格名 + 图片文件名
            writer.writerow([style, fname])

print(f"Annotations CSV generated at: {CSV_PATH}")
