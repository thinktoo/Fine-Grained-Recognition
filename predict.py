"""
WebFG-400 模型预测脚本
功能：
1. 加载训练好的模型进行预测
2. 正确映射预测索引到原始类别编号
3. 支持单张图片预测和批量预测
4. 生成提交文件和预测报告
5. 提供Top-K预测和置信度分析
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import time
import argparse

from model import ImprovedAIModel
from config import Config, get_class_mapping, TestDataset


class WebFG400Predictor:
    """WebFG-400 预测器类"""

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化预测器

        Args:
            model_path (str): 训练好的模型路径
            device (str): 推理设备
        """
        self.device = device
        self.num_classes = Config.NUM_CLASSES

        print(f"🔧 初始化WebFG-400预测器...")
        print(f"   设备: {self.device}")
        print(f"   类别数: {self.num_classes}")

        # 创建类别映射
        self.class_mapping = self._create_class_mapping()
        print(f"✅ 类别映射创建完成")

        # 加载模型
        self.model = self._load_model(model_path)
        print(f"✅ 模型加载完成")

        # 图像预处理
        self.transform = self._get_transform()

    def _create_class_mapping(self):
        """创建类别映射"""
        class_mapping = get_class_mapping()
        self.index_to_class = class_mapping['train_to_original']
        self.class_to_index = class_mapping['original_to_train']

        print(f"   映射示例: 索引223→类别'{self.index_to_class[223]}', 索引224→类别'{self.index_to_class[224]}'")
        return class_mapping

    def _load_model(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

        print(f"📥 加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # 获取模型配置
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"   模型配置: {config}")
        else:
            # 使用默认配置
            config = {
                'backbone': Config.BACKBONE,
                'embed_dim': Config.EMBED_DIM,
                'num_heads': Config.NUM_HEADS,
                'num_classes': self.num_classes,
                'num_coarse_classes': Config.NUM_COARSE_CLASSES
            }

        # 创建模型
        model = ImprovedAIModel(
            arch=config.get('backbone', Config.BACKBONE),
            num_classes=config.get('num_classes', self.num_classes),
            num_coarse_classes=config.get('num_coarse_classes', Config.NUM_COARSE_CLASSES),
            embed_dim=config.get('embed_dim', Config.EMBED_DIM),
            num_heads=config.get('num_heads', Config.NUM_HEADS),
            top_ratio=Config.TOP_RATIO,
            num_sa_layers=Config.NUM_SA_LAYERS,
            num_pwca_layers=Config.NUM_PWCA_LAYERS
        )

        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   模型参数量: {total_params:,}")

        if 'best_ensemble_acc' in checkpoint:
            print(f"   最佳验证准确率: {checkpoint['best_ensemble_acc']:.4f}")

        return model

    def _get_transform(self):
        """获取图像预处理变换"""
        return transforms.Compose([
            transforms.Resize((Config.RESIZE_SIZE, Config.RESIZE_SIZE)),
            transforms.CenterCrop(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_single_image(self, image_path, return_features=False):
        """
        预测单张图片

        Args:
            image_path (str): 图像路径
            return_features (bool): 是否返回特征向量

        Returns:
            dict: 预测结果
        """
        try:
            # 加载并预处理图片
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 模型推理
            with torch.no_grad():
                outputs = self.model(input_tensor)

                # 使用GLCA分支的输出作为最终预测
                logits = outputs['glca_logits']
                probabilities = F.softmax(logits, dim=1)

                # 获取预测结果
                prediction_index = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction_index].item()

            # 映射回原始类别编号
            predicted_class = self.index_to_class[prediction_index]

            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'prediction_index': prediction_index,
                'probabilities': probabilities[0].cpu().numpy() if return_features else None
            }

            # 添加辅助信息
            if 'sa_logits' in outputs and outputs['sa_logits'] is not None:
                sa_probs = F.softmax(outputs['sa_logits'], dim=1)
                sa_pred = torch.argmax(sa_probs, dim=1).item()
                result['sa_prediction'] = sa_pred
                result['sa_confidence'] = sa_probs[0][sa_pred].item()

            return result

        except Exception as e:
            return {'error': str(e)}

    def predict_batch(self, image_folder, batch_size=32, save_results=True):
        """
        批量预测文件夹中的所有图片

        Args:
            image_folder (str): 图像文件夹路径
            batch_size (int): 批次大小
            save_results (bool): 是否保存结果

        Returns:
            list: 预测结果列表
        """
        print(f"🔮 开始批量预测...")
        print(f"   图像目录: {image_folder}")
        print(f"   批次大小: {batch_size}")

        # 创建测试数据集
        test_dataset = TestDataset(image_folder, transform=self.transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        results = []
        total_time = 0

        with torch.no_grad():
            for batch_images, batch_filenames in tqdm(test_loader, desc="预测批次"):
                batch_start_time = time.time()

                batch_images = batch_images.to(self.device)

                # 批量推理
                outputs = self.model(batch_images)
                logits = outputs['glca_logits']
                probabilities = F.softmax(logits, dim=1)

                # 处理批次结果
                for i, filename in enumerate(batch_filenames):
                    pred_index = torch.argmax(probabilities[i]).item()
                    confidence = probabilities[i][pred_index].item()
                    predicted_class = self.index_to_class[pred_index]

                    results.append({
                        'filename': filename,
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'prediction_index': pred_index
                    })

                batch_time = time.time() - batch_start_time
                total_time += batch_time

        # 统计信息
        avg_time_per_image = total_time / len(results) if len(results) > 0 else 0
        print(f"✅ 批量预测完成!")
        print(f"   处理图像: {len(results)}")
        print(f"   总耗时: {total_time:.2f}s")
        print(f"   平均每张: {avg_time_per_image * 1000:.1f}ms")

        # 保存结果
        if save_results:
            self._save_prediction_results(results, image_folder)

        return results

    def get_top_k_predictions(self, image_path, k=5):
        """
        获取Top-K预测结果

        Args:
            image_path (str): 图像路径
            k (int): 返回前K个预测

        Returns:
            list: Top-K预测结果
        """
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs['glca_logits']
                probabilities = F.softmax(logits, dim=1)

                # 获取Top-K结果
                top_k_probs, top_k_indices = torch.topk(probabilities, k, dim=1)

            results = []
            for i in range(k):
                pred_idx = top_k_indices[0][i].item()
                prob = top_k_probs[0][i].item()
                class_name = self.index_to_class[pred_idx]

                results.append({
                    'rank': i + 1,
                    'predicted_class': class_name,
                    'confidence': prob,
                    'prediction_index': pred_idx
                })

            return results

        except Exception as e:
            return {'error': str(e)}

    def create_submission_file(self, test_folder, output_path=None):
        """
        创建比赛提交文件

        Args:
            test_folder (str): 测试图像文件夹
            output_path (str): 输出文件路径

        Returns:
            str: 生成的提交文件路径
        """
        if output_path is None:
            output_path = Config.SUBMISSION_PATH

        print(f"📝 创建提交文件...")

        # 批量预测
        results = self.predict_batch(test_folder, save_results=False)

        # 按文件名排序（重要！）
        results.sort(key=lambda x: x['filename'])

        # 创建CSV文件
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_name', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow({
                    'image_name': result['filename'],
                    'label': result['predicted_class']
                })

        print(f"✅ 提交文件创建完成: {output_path}")
        print(f"   总预测数: {len(results)}")

        # 验证文件格式
        self._validate_submission_file(output_path)

        return output_path

    def _validate_submission_file(self, submission_path):
        """验证提交文件格式"""
        try:
            df = pd.read_csv(submission_path)

            # 检查列名
            expected_columns = ['image_name', 'label']
            if list(df.columns) != expected_columns:
                print(f"⚠️  警告: 列名不正确. 期望: {expected_columns}, 实际: {list(df.columns)}")

            # 检查数据
            print(f"📋 提交文件验证:")
            print(f"   行数: {len(df)}")
            print(f"   列数: {len(df.columns)}")
            print(f"   缺失值: {df.isnull().sum().sum()}")

            # 显示前几行
            print(f"   前5行预览:")
            print(df.head().to_string(index=False))

            # 检查标签格式
            unique_labels = df['label'].unique()
            print(f"   唯一标签数: {len(unique_labels)}")
            print(f"   标签示例: {sorted(unique_labels)[:10]}")

        except Exception as e:
            print(f"❌ 提交文件验证失败: {e}")

    def _save_prediction_results(self, results, image_folder):
        """保存预测结果到JSON文件"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = f"prediction_results_{timestamp}.json"

        # 添加统计信息
        stats = self._compute_prediction_stats(results)

        output_data = {
            'metadata': {
                'timestamp': timestamp,
                'image_folder': image_folder,
                'total_images': len(results),
                'model_info': {
                    'device': self.device,
                    'num_classes': self.num_classes
                }
            },
            'statistics': stats,
            'predictions': results
        }

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"📄 预测结果已保存: {results_path}")

    def _compute_prediction_stats(self, results):
        """计算预测统计信息"""
        if not results:
            return {}

        confidences = [r['confidence'] for r in results]
        predicted_classes = [r['predicted_class'] for r in results]

        # 置信度统计
        confidence_stats = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'median': np.median(confidences)
        }

        # 类别分布统计
        from collections import Counter
        class_counts = Counter(predicted_classes)
        most_common_classes = class_counts.most_common(10)

        return {
            'confidence_stats': confidence_stats,
            'class_distribution': {
                'total_unique_classes': len(class_counts),
                'most_common_classes': most_common_classes
            }
        }

    def analyze_predictions(self, results, top_k=10):
        """分析预测结果"""
        print(f"\n📊 预测结果分析:")
        print(f"   总预测数: {len(results)}")

        if not results:
            return

        # 置信度分析
        confidences = [r['confidence'] for r in results if 'confidence' in r]
        if confidences:
            print(f"   置信度统计:")
            print(f"     平均值: {np.mean(confidences):.4f}")
            print(f"     标准差: {np.std(confidences):.4f}")
            print(f"     最小值: {np.min(confidences):.4f}")
            print(f"     最大值: {np.max(confidences):.4f}")

            # 置信度分布
            high_conf = sum(1 for c in confidences if c > 0.8)
            medium_conf = sum(1 for c in confidences if 0.5 < c <= 0.8)
            low_conf = sum(1 for c in confidences if c <= 0.5)

            print(f"   置信度分布:")
            print(f"     高置信度(>0.8): {high_conf} ({high_conf / len(confidences) * 100:.1f}%)")
            print(f"     中等置信度(0.5-0.8): {medium_conf} ({medium_conf / len(confidences) * 100:.1f}%)")
            print(f"     低置信度(≤0.5): {low_conf} ({low_conf / len(confidences) * 100:.1f}%)")

        # 类别分布分析
        predicted_classes = [r['predicted_class'] for r in results if 'predicted_class' in r]
        if predicted_classes:
            from collections import Counter
            class_counts = Counter(predicted_classes)

            print(f"   类别分布:")
            print(f"     预测的唯一类别数: {len(class_counts)}")
            print(f"     最常预测的{top_k}个类别:")

            for i, (class_name, count) in enumerate(class_counts.most_common(top_k)):
                percentage = count / len(predicted_classes) * 100
                print(f"       {i + 1:2d}. {class_name}: {count:4d} ({percentage:5.1f}%)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='WebFG-400模型预测脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 创建提交文件
  python predict.py --model model/best_model.pth --test webfg400_test_A --output submission.csv

  # 单张图片预测
  python predict.py --model model/best_model.pth --image test.jpg --top_k 5

  # 批量预测并分析
  python predict.py --model model/best_model.pth --test test_folder --analyze
        """
    )

    parser.add_argument('--model', required=True, help='训练好的模型路径')
    parser.add_argument('--test', help='测试图像文件夹路径')
    parser.add_argument('--image', help='单张图像路径')
    parser.add_argument('--output', default='submission.csv', help='提交文件输出路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批量预测的批次大小')
    parser.add_argument('--top_k', type=int, default=5, help='显示Top-K预测结果')
    parser.add_argument('--analyze', action='store_true', help='分析预测结果')
    parser.add_argument('--device', default='auto', help='推理设备 (cuda/cpu/auto)')

    args = parser.parse_args()

    # 设备选择
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("🚀 WebFG-400 模型预测")
    print("=" * 50)
    print(f"📂 模型路径: {args.model}")
    print(f"💻 设备: {device}")

    try:
        # 创建预测器
        predictor = WebFG400Predictor(args.model, device=device)

        if args.image:
            # 单张图片预测
            print(f"\n🖼️  单张图片预测: {args.image}")

            if not os.path.exists(args.image):
                print(f"❌ 图片文件不存在: {args.image}")
                return

            # 标准预测
            result = predictor.predict_single_image(args.image)
            if 'error' in result:
                print(f"❌ 预测失败: {result['error']}")
                return

            print(f"✅ 预测结果:")
            print(f"   预测类别: {result['predicted_class']}")
            print(f"   置信度: {result['confidence']:.4f}")

            # Top-K预测
            if args.top_k > 1:
                print(f"\n📊 Top-{args.top_k} 预测结果:")
                top_k_results = predictor.get_top_k_predictions(args.image, args.top_k)

                if 'error' not in top_k_results:
                    for result in top_k_results:
                        print(f"   {result['rank']:2d}. {result['predicted_class']} "
                              f"(置信度: {result['confidence']:.4f})")

        elif args.test:
            # 批量预测
            print(f"\n📁 批量预测: {args.test}")

            if not os.path.exists(args.test):
                print(f"❌ 测试目录不存在: {args.test}")
                return

            # 创建提交文件
            submission_path = predictor.create_submission_file(args.test, args.output)

            # 如果需要分析结果
            if args.analyze:
                print(f"\n📈 开始结果分析...")
                results = predictor.predict_batch(args.test, args.batch_size, save_results=False)
                predictor.analyze_predictions(results, top_k=10)

        else:
            # 默认行为：对测试集进行预测
            test_folder = Config.TEST_DATA_PATH
            if os.path.exists(test_folder):
                print(f"\n📁 使用默认测试目录: {test_folder}")
                predictor.create_submission_file(test_folder, args.output)
            else:
                print(f"❌ 请指定 --test 或 --image 参数")
                parser.print_help()

    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 预测过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
