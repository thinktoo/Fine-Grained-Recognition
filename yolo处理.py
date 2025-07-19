"""
使用YOLO检测并保留包含飞机、车辆、鸟类的图像，删除其他无关图像
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# YOLO检测的相关类别ID (COCO数据集)
RELEVANT_CLASSES = {
    'airplane': [4],  # airplane
    'car': [2, 3, 5, 6, 7],  # bicycle, car, motorcycle, bus, truck
    'bird': [14],  # bird
}

# 将所有相关类别ID合并
KEEP_CLASS_IDS = []
for category, class_ids in RELEVANT_CLASSES.items():
    KEEP_CLASS_IDS.extend(class_ids)

print(f"保留的YOLO类别ID: {KEEP_CLASS_IDS}")


def detect_relevant_objects(image_path, model, confidence_threshold=0.3):
    """
    检测图像中是否包含相关对象（飞机、车、鸟）

    Args:
        image_path: 图像路径
        model: YOLO模型
        confidence_threshold: 置信度阈值

    Returns:
        bool: 是否包含相关对象
    """
    try:
        # YOLO检测
        results = model(image_path, conf=confidence_threshold, verbose=False)

        # 检查检测结果
        for result in results:
            if result.boxes is not None:
                # 获取检测到的类别
                detected_classes = result.boxes.cls.cpu().numpy()

                # 检查是否包含相关类别
                for class_id in detected_classes:
                    if int(class_id) in KEEP_CLASS_IDS:
                        return True

        return False

    except Exception as e:
        print(f"检测失败 {image_path}: {e}")
        return False


def clean_dataset(root_dir, output_dir, confidence_threshold=0.3):
    """
    清洗数据集，保留包含相关对象的图像

    Args:
        root_dir: 原始数据集目录
        output_dir: 清洗后数据集目录
        confidence_threshold: YOLO检测置信度阈值
    """
    # 加载YOLO模型
    print("加载YOLO模型...")
    model = YOLO('yolov8n.pt')  # 使用nano版本，速度快

    # 数据集路径 - 修改这里
    train_path = os.path.join(root_dir, 'webfg400_train_dirty', 'train')  # 源数据
    output_path = os.path.join(output_dir, 'webfg400_train', 'train')  # 目标数据

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练数据目录未找到: {train_path}")

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 统计信息
    total_images = 0
    kept_images = 0
    removed_images = 0

    # 遍历所有类别目录
    for class_name in os.listdir(train_path):
        class_dir = os.path.join(train_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"\n处理类别: {class_name}")

        # 创建输出类别目录
        output_class_dir = os.path.join(output_path, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        # 获取该类别的所有图像
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(Path(class_dir).glob(f'*{ext}'))
            image_files.extend(Path(class_dir).glob(f'*{ext.upper()}'))

        class_total = len(image_files)
        class_kept = 0

        # 处理每张图像
        for img_path in tqdm(image_files, desc=f"处理 {class_name}"):
            total_images += 1

            # YOLO检测
            if detect_relevant_objects(str(img_path), model, confidence_threshold):
                # 保留图像
                output_img_path = os.path.join(output_class_dir, img_path.name)
                shutil.copy2(str(img_path), output_img_path)
                kept_images += 1
                class_kept += 1
            else:
                # 删除图像（实际上是不复制到新目录）
                removed_images += 1

        print(f"  类别 {class_name}: {class_kept}/{class_total} 张图像被保留")

        # 如果某个类别没有任何图像被保留，删除该类别目录
        if class_kept == 0:
            shutil.rmtree(output_class_dir)
            print(f"  类别 {class_name} 被完全删除（无相关图像）")

    # 打印总结
    print(f"\n=== 数据清洗完成 ===")
    print(f"总图像数: {total_images}")
    print(f"保留图像: {kept_images} ({kept_images / total_images * 100:.1f}%)")
    print(f"移除图像: {removed_images} ({removed_images / total_images * 100:.1f}%)")
    print(f"清洗后数据集保存在: {output_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='使用YOLO清洗细粒度分类数据集')
    parser.add_argument('--root', default='.', help='原始数据集根目录')
    parser.add_argument('--output', default='./cleaned_data', help='清洗后数据集输出目录')
    parser.add_argument('--confidence', type=float, default=0.3, help='YOLO检测置信度阈值')
    parser.add_argument('--dry_run', action='store_true', help='只统计不实际操作')

    args = parser.parse_args()

    print("=== 细粒度数据集清洗工具 ===")
    print(f"原始数据目录: {args.root}")
    print(f"输出目录: {args.output}")
    print(f"置信度阈值: {args.confidence}")
    print(f"保留对象: 飞机、车辆、鸟类")

    if args.dry_run:
        print("\n[DRY RUN模式] 仅统计，不实际操作")

    # 开始清洗
    clean_dataset(args.root, args.output, args.confidence)


if __name__ == '__main__':
    main()
