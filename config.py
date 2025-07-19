"""
WebFG-400 配置和数据处理模块
功能：
1. 统一配置管理
2. 数据集加载和预处理
3. 类别映射处理
4. 训练/验证数据集划分
"""

import os
import torch
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from ultralytics import YOLO
import json
import random

# 允许加载截断的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Config:
    """统一配置参数类"""

    # === 数据相关参数 ===
    IMAGE_SIZE = 448                # 输入图像尺寸
    RESIZE_SIZE = 512              # 预处理时的缩放尺寸
    TRAIN_VAL_SPLIT = 0.8          # 训练/验证集划分比例
    RANDOM_SEED = 42               # 随机种子

    # === 模型架构参数 ===
    BACKBONE = 'efficientnet-b6'   # 骨干网络
    EMBED_DIM = 512                # 嵌入维度
    NUM_HEADS = 16                 # 多头注意力头数
    TOP_RATIO = 0.15               # GLCA模块选择top patches的比例
    NUM_SA_LAYERS = 8              # 自注意力层数
    NUM_PWCA_LAYERS = 6            # 成对交叉注意力层数
    DROPOUT_RATE = 0.2             # Dropout比例

    # === 训练参数 (4090 24G标准配置) ===
    BATCH_SIZE = 40                # 批次大小 (24G显存标准值)
    LEARNING_RATE = 1.2e-3         # 学习率
    WEIGHT_DECAY = 5e-4            # 权重衰减
    EPOCHS = 100                   # 训练轮数
    WARMUP_EPOCHS = 8              # 预热轮数

    # === 高级训练技巧 ===
    USE_AMP = True                 # 混合精度训练
    USE_MIXUP = True               # MixUp数据增强
    USE_CUTMIX = True              # CutMix数据增强
    USE_LABEL_SMOOTHING = True     # 标签平滑
    MIXUP_ALPHA = 0.3              # MixUp参数
    CUTMIX_ALPHA = 1.2             # CutMix参数

    # === 损失函数参数 ===
    FOCAL_ALPHA = 1.0              # Focal Loss的alpha参数
    FOCAL_GAMMA = 2.0              # Focal Loss的gamma参数
    LABEL_SMOOTHING = 0.15         # 标签平滑参数

    # === 层次化训练参数 ===
    SA_WEIGHT = 0.4                # 自注意力分支损失权重
    GLCA_WEIGHT = 1.0              # GLCA分支损失权重
    NUM_COARSE_CLASSES = 3         # 大类数量（飞机、车辆、鸟类）

    # === YOLO检测参数 ===
    YOLO_MODEL = 'yolov8s.pt'      # YOLO模型版本 (small版本平衡速度和精度)
    YOLO_CONFIDENCE = 0.3          # YOLO检测置信度阈值
    YOLO_AIRPLANE_IDS = [4]        # YOLO中飞机类别ID
    YOLO_VEHICLE_IDS = [2, 3, 5, 6, 7]  # YOLO中车辆类别ID
    YOLO_BIRD_IDS = [14]           # YOLO中鸟类类别ID

    # === 文件路径 ===
    DATA_ROOT = '.'                # 数据根目录
    TRAIN_DATA_PATH = 'webfg400_train/train'  # 训练数据路径
    TEST_DATA_PATH = 'webfg400_test_A'        # 测试数据路径
    MODEL_SAVE_PATH = 'model/best_model.pth'  # 模型保存路径
    SUBMISSION_PATH = 'submission.csv'         # 提交文件路径

    # === 设备配置 (4090 24G优化) ===
    DEVICE = 'cuda'                # 训练设备
    NUM_WORKERS = 12               # 数据加载器工作进程数
    PIN_MEMORY = True              # 是否使用固定内存
    PREFETCH_FACTOR = 4            # 预取因子

    # === 类别映射参数 ===
    NUM_CLASSES = 399              # 排除224后的总类别数


# 全局变量，用于缓存数据集和映射
_train_subset = None
_val_subset = None
_class_mapping = None



def create_class_mapping():
    """
    创建类别映射：训练索引 ↔ 原始类别编号

    关键映射逻辑：
    - 0-223 保持不变
    - 跳过224
    - 225-399 映射到224-398
    """
    global _class_mapping
    if _class_mapping is not None:
        return _class_mapping

    # 创建双向映射
    original_to_train = {}  # 原始类别编号(字符串) → 训练标签(整数)
    train_to_original = {}  # 训练标签(整数) → 原始类别编号(字符串)

    train_idx = 0
    for i in range(400):  # 遍历原始类别 0-399
        if i != 224:  # 跳过224类
            class_name = f"{i:03d}"  # 格式化为三位数字符串
            original_to_train[class_name] = train_idx
            train_to_original[train_idx] = class_name
            train_idx += 1

    _class_mapping = {
        'original_to_train': original_to_train,
        'train_to_original': train_to_original
    }

    print(f"✅ 类别映射创建完成，共 {len(train_to_original)} 个有效类别")
    return _class_mapping


class MixupCutmix:
    """MixUp和CutMix数据增强实现"""

    def __init__(self, mixup_alpha=0.3, cutmix_alpha=1.2, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def __call__(self, x, y):
        if torch.rand(1) < self.prob:
            if torch.rand(1) < 0.5:
                return self.mixup(x, y)
            else:
                return self.cutmix(x, y)
        return x, y, y, 1.0

    def mixup(self, x, y):
        """MixUp实现：线性混合图片和标签"""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, y, y[index], lam

    def cutmix(self, x, y):
        """CutMix实现：剪切并粘贴图片区域"""
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        H, W = x.size(2), x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        return x, y, y[index], lam


class FilteredImageFolder(ImageFolder):
    """过滤掉224类别的ImageFolder"""

    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)

        # 创建类别映射
        class_mapping = create_class_mapping()

        # 过滤掉224类别的样本
        filtered_samples = []
        for path, original_class_idx in self.samples:
            class_name = self.classes[original_class_idx]

            if class_name != '224':  # 跳过224类
                new_class_idx = class_mapping['original_to_train'][class_name]
                filtered_samples.append((path, new_class_idx))

        self.samples = filtered_samples

        # 创建新的类别列表
        self.filtered_classes = []
        train_to_original = class_mapping['train_to_original']
        for train_idx in range(Config.NUM_CLASSES):
            original_class = train_to_original[train_idx]
            self.filtered_classes.append(original_class)

        self.classes = self.filtered_classes
        self.class_to_idx = class_mapping['original_to_train']

        print(f"✅ 过滤完成: 剩余 {len(self.samples)} 个样本，{len(self.classes)} 个类别")


class ImageFolderSubset(Dataset):
    """ImageFolder的子集包装器"""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def detect_object_type(image_path):
    """使用YOLO检测图像中的对象类型"""
    try:
        yolo_model = YOLO(Config.YOLO_MODEL)
        results = yolo_model(image_path, conf=Config.YOLO_CONFIDENCE, verbose=False)

        for result in results:
            if result.boxes is not None:
                detected_classes = result.boxes.cls.cpu().numpy()

                for class_id in detected_classes:
                    class_id = int(class_id)
                    if class_id in Config.YOLO_AIRPLANE_IDS:
                        return 0  # 飞机
                    elif class_id in Config.YOLO_VEHICLE_IDS:
                        return 1  # 车辆
                    elif class_id in Config.YOLO_BIRD_IDS:
                        return 2  # 鸟类

        return 2  # 默认鸟类

    except Exception as e:
        print(f"YOLO检测失败 {image_path}: {e}")
        return 2


def create_coarse_mapping(class_names):
    """创建或加载大类映射"""
    try:
        with open('coarse_mapping.json', 'r') as f:
            coarse_mapping = json.load(f)
        return {int(k): v for k, v in coarse_mapping.items()}
    except FileNotFoundError:
        print("⚠️  没有找到大类映射缓存文件，返回空字典")
        return {}


def _prepare_datasets(root_dir):
    """准备训练和验证数据集"""
    global _train_subset, _val_subset
    if _train_subset is not None and _val_subset is not None:
        return

    base_dir = os.path.join(root_dir, Config.TRAIN_DATA_PATH)
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"训练数据目录未找到: {base_dir}")

    full_dataset = FilteredImageFolder(base_dir)

    # 划分训练和验证集
    n_total = len(full_dataset.samples)
    train_len = int(n_total * Config.TRAIN_VAL_SPLIT)

    generator = torch.Generator().manual_seed(Config.RANDOM_SEED)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    # 创建不同变换的数据集
    train_set = FilteredImageFolder(base_dir, transform=get_train_transform())
    val_set = FilteredImageFolder(base_dir, transform=get_val_transform())

    _train_subset = ImageFolderSubset(train_set, train_indices)
    _val_subset = ImageFolderSubset(val_set, val_indices)


def get_train_transform():
    """训练时的数据增强变换"""
    return transforms.Compose([
        transforms.Resize((Config.RESIZE_SIZE, Config.RESIZE_SIZE)),
        transforms.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.75, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.25)),
    ])


def get_val_transform():
    """验证时的数据变换"""
    return transforms.Compose([
        transforms.Resize((Config.RESIZE_SIZE, Config.RESIZE_SIZE)),
        transforms.CenterCrop(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_train_dataset(root_dir):
    """获取训练数据集"""
    _prepare_datasets(root_dir)
    return _train_subset


def get_val_dataset(root_dir):
    """获取验证数据集"""
    _prepare_datasets(root_dir)
    return _val_subset


def get_class_mapping():
    """获取类别映射"""
    return create_class_mapping()


class TestDataset(Dataset):
    """测试数据集类"""

    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform if transform else get_val_transform()

        self.image_files = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

        for file in os.listdir(root_dir):
            if file.lower().endswith(valid_extensions):
                self.image_files.append(file)

        self.image_files.sort()
        print(f"✅ 测试数据集加载完成，共 {len(self.image_files)} 张图片")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        path = os.path.join(self.root, fname)
        img = Image.open(path).convert('RGB')
        return self.transform(img), fname
