"""
WebFG-400 层次化训练脚本
功能：
1. 层次化模型训练（SA + GLCA + PWCA）
2. 混合精度训练优化
3. 高级学习率调度
4. MixUp/CutMix数据增强
5. 模型检查点管理
"""

import os
import torch
import math
import time
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from config import Config, get_train_dataset, get_val_dataset, create_coarse_mapping, MixupCutmix
from model import ImprovedAIModel, FocalLoss


def get_advanced_scheduler(optimizer, epochs, warmup_epochs):
    """余弦退火+warmup学习率调度器"""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp损失计算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, coarse_mapping):
    """训练一个epoch"""
    model.train()
    mixup_cutmix = MixupCutmix(Config.MIXUP_ALPHA, Config.CUTMIX_ALPHA, prob=0.5)

    total_loss = 0.0
    total_sa_loss = 0.0
    total_glca_loss = 0.0
    num_samples = 0

    batch_time = 0.0
    data_time = 0.0
    end = time.time()

    for batch_idx, (img, fine_label) in enumerate(loader):
        data_time += time.time() - end

        img = img.to(device, non_blocking=True)
        fine_label = fine_label.to(device, non_blocking=True)

        # MixUp/CutMix数据增强
        use_mixup = False
        if (Config.USE_MIXUP or Config.USE_CUTMIX) and model.training:
            img, fine_label_a, fine_label_b, lam = mixup_cutmix(img, fine_label)

            coarse_label_a = torch.tensor([coarse_mapping[l.item()] for l in fine_label_a],
                                          device=device, dtype=torch.long)
            coarse_label_b = torch.tensor([coarse_mapping[l.item()] for l in fine_label_b],
                                          device=device, dtype=torch.long)
            use_mixup = True
        else:
            coarse_label = torch.tensor([coarse_mapping[l.item()] for l in fine_label],
                                        device=device, dtype=torch.long)

        # 混合精度前向传播
        with autocast():
            if model.training and torch.rand(1) < 0.5:
                shuffle_idx = torch.randperm(img.size(0), device=device)
                img2 = img[shuffle_idx]
                outputs = model(img, img2)
            else:
                outputs = model(img)

            # 计算损失
            if use_mixup:
                loss_sa = mixup_criterion(criterion, outputs['sa_logits'],
                                          coarse_label_a, coarse_label_b, lam)
                loss_glca = mixup_criterion(criterion, outputs['glca_logits'],
                                            fine_label_a, fine_label_b, lam)
            else:
                loss_sa = criterion(outputs['sa_logits'], coarse_label)
                loss_glca = criterion(outputs['glca_logits'], fine_label)

            total_loss_batch = Config.SA_WEIGHT * loss_sa + Config.GLCA_WEIGHT * loss_glca

        # 反向传播
        optimizer.zero_grad()
        scaler.scale(total_loss_batch).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()

        # 统计
        batch_size = img.size(0)
        total_loss += total_loss_batch.item() * batch_size
        total_sa_loss += loss_sa.item() * batch_size
        total_glca_loss += loss_glca.item() * batch_size
        num_samples += batch_size

        batch_time += time.time() - end
        end = time.time()

        # 进度输出
        if batch_idx % 50 == 0:
            avg_batch_time = batch_time / (batch_idx + 1)
            avg_data_time = data_time / (batch_idx + 1)

            print(f'Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(loader):4d} | '
                  f'Loss: {total_loss_batch.item():.4f} | '
                  f'SA: {loss_sa.item():.4f} | '
                  f'GLCA: {loss_glca.item():.4f} | '
                  f'Time: {avg_batch_time:.3f}s | '
                  f'Data: {avg_data_time:.3f}s | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    return total_loss / num_samples, total_sa_loss / num_samples, total_glca_loss / num_samples


def validate(model, loader, criterion, device, coarse_mapping):
    """验证模型性能"""
    model.eval()

    total_loss = 0.0
    correct_sa = 0
    correct_glca = 0
    correct_ensemble = 0
    total_samples = 0

    with torch.no_grad():
        for img, fine_label in loader:
            img = img.to(device, non_blocking=True)
            fine_label = fine_label.to(device, non_blocking=True)

            coarse_label = torch.tensor([coarse_mapping[label.item()] for label in fine_label],
                                        device=device, dtype=torch.long)

            with autocast():
                outputs = model(img)
                loss_sa = criterion(outputs['sa_logits'], coarse_label)
                loss_glca = criterion(outputs['glca_logits'], fine_label)
                batch_loss = Config.SA_WEIGHT * loss_sa + Config.GLCA_WEIGHT * loss_glca

            total_loss += batch_loss.item() * img.size(0)

            pred_sa = outputs['sa_logits'].argmax(dim=1)
            pred_glca = outputs['glca_logits'].argmax(dim=1)

            correct_sa += (pred_sa == coarse_label).sum().item()
            correct_glca += (pred_glca == fine_label).sum().item()
            correct_ensemble += (pred_glca == fine_label).sum().item()

            total_samples += img.size(0)

    avg_loss = total_loss / total_samples
    sa_acc = correct_sa / total_samples
    glca_acc = correct_glca / total_samples
    ensemble_acc = correct_ensemble / total_samples

    return avg_loss, sa_acc, glca_acc, ensemble_acc


def setup_training():
    """设置训练环境"""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_acc, coarse_mapping, filepath):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_ensemble_acc': best_acc,
        'coarse_mapping': coarse_mapping,
        'config': {
            'backbone': Config.BACKBONE,
            'embed_dim': Config.EMBED_DIM,
            'num_heads': Config.NUM_HEADS,
            'num_classes': Config.NUM_CLASSES,
            'num_coarse_classes': Config.NUM_COARSE_CLASSES,
            'image_size': Config.IMAGE_SIZE,
            'batch_size': Config.BATCH_SIZE,
        }
    }

    torch.save(checkpoint, filepath)
    print(f'💾 检查点已保存: {filepath}')


def load_checkpoint(model, optimizer, scheduler, scaler, filepath, device):
    """加载训练检查点"""
    try:
        print(f"🔄 加载检查点: {filepath}")
        checkpoint = torch.load(filepath, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_ensemble_acc', 0.0)
        coarse_mapping = checkpoint.get('coarse_mapping', {})

        print(f"✅ 检查点加载成功!")
        print(f"   恢复到第 {start_epoch} 轮")
        print(f"   当前最佳准确率: {best_acc:.4f}")

        return start_epoch, best_acc, coarse_mapping

    except Exception as e:
        print(f"❌ 检查点加载失败: {e}")
        print("从头开始训练...")
        return 1, 0.0, {}


def print_model_info(model, device):
    """打印模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n🏗️  模型信息:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")

    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)
        outputs = model(dummy_input)
        print(f"   SA输出形状: {outputs['sa_logits'].shape}")
        print(f"   GLCA输出形状: {outputs['glca_logits'].shape}")

    model.train()


def main():
    """主训练函数"""
    print("🚀 WebFG-400 层次化训练开始")
    print("=" * 60)
    print(f"📊 训练配置:")
    print(f"   模型骨干: {Config.BACKBONE}")
    print(f"   图像尺寸: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    print(f"   批次大小: {Config.BATCH_SIZE}")
    print(f"   学习率: {Config.LEARNING_RATE}")
    print(f"   训练轮数: {Config.EPOCHS}")
    print(f"   SA权重: {Config.SA_WEIGHT}, GLCA权重: {Config.GLCA_WEIGHT}")
    print(f"   混合精度: {'✅' if Config.USE_AMP else '❌'}")
    print(f"   MixUp: {'✅' if Config.USE_MIXUP else '❌'}")
    print(f"   CutMix: {'✅' if Config.USE_CUTMIX else '❌'}")

    setup_training()

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 设备信息:")
    print(f"   训练设备: {device}")

    if device.type == 'cuda':
        print(f"   GPU名称: {torch.cuda.get_device_name()}")
        print(f"   显存容量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)

    # 数据集加载
    print(f"\n📂 数据集加载...")
    train_set = get_train_dataset(Config.DATA_ROOT)
    val_set = get_val_dataset(Config.DATA_ROOT)

    train_loader = DataLoader(
        train_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=True
    )

    num_classes = len(val_set.classes)
    print(f"✅ 数据集加载完成:")
    print(f"   训练样本: {len(train_set):,}")
    print(f"   验证样本: {len(val_set):,}")
    print(f"   类别数量: {num_classes}")

    # 创建大类映射
    print(f"\n🔍 创建大类映射...")
    coarse_mapping = create_coarse_mapping(val_set.classes)
    if not coarse_mapping:
        print("❌ 大类映射创建失败，请检查数据集")
        return

    # 模型初始化
    print(f"\n🏗️  模型初始化...")
    model = ImprovedAIModel(
        arch=Config.BACKBONE,
        num_classes=num_classes,
        num_coarse_classes=Config.NUM_COARSE_CLASSES,
        embed_dim=Config.EMBED_DIM,
        num_heads=Config.NUM_HEADS,
        top_ratio=Config.TOP_RATIO,
        num_sa_layers=Config.NUM_SA_LAYERS,
        num_pwca_layers=Config.NUM_PWCA_LAYERS
    ).to(device)

    print_model_info(model, device)

    # 损失函数和优化器
    criterion = FocalLoss(
        alpha=Config.FOCAL_ALPHA,
        gamma=Config.FOCAL_GAMMA,
        label_smoothing=Config.LABEL_SMOOTHING
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler = get_advanced_scheduler(optimizer, Config.EPOCHS, Config.WARMUP_EPOCHS)
    scaler = GradScaler()

    # 检查点恢复
    start_epoch = 1
    best_ensemble_acc = 0.0

    if os.path.exists(Config.MODEL_SAVE_PATH):
        start_epoch, best_ensemble_acc, loaded_coarse_mapping = load_checkpoint(
            model, optimizer, scheduler, scaler, Config.MODEL_SAVE_PATH, device
        )
        if loaded_coarse_mapping:
            coarse_mapping = loaded_coarse_mapping

    # 训练循环
    print(f"\n🔥 开始训练循环...")
    print("=" * 80)

    train_start_time = time.time()

    for epoch in range(start_epoch, Config.EPOCHS + 1):
        epoch_start_time = time.time()

        print(f'\n{"=" * 20} Epoch {epoch:3d}/{Config.EPOCHS} {"=" * 20}')

        # 训练
        train_loss, train_sa_loss, train_glca_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, coarse_mapping
        )

        # 验证
        val_loss, val_sa_acc, val_glca_acc, val_ensemble_acc = validate(
            model, val_loader, criterion, device, coarse_mapping
        )

        scheduler.step()

        # 计算时间
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - train_start_time

        # 输出结果
        print(f'\n📊 Epoch {epoch} 结果:')
        print(f'   训练 - 总损失: {train_loss:.4f} | SA: {train_sa_loss:.4f} | GLCA: {train_glca_loss:.4f}')
        print(
            f'   验证 - 损失: {val_loss:.4f} | SA准确率: {val_sa_acc:.4f} | GLCA准确率: {val_glca_acc:.4f} | 集成准确率: {val_ensemble_acc:.4f}')
        print(f'   学习率: {scheduler.get_last_lr()[0]:.8f}')
        print(f'   训练时间: {epoch_time:.1f}s | 总时间: {total_time / 3600:.1f}h')

        # 保存最佳模型
        if val_ensemble_acc > best_ensemble_acc:
            best_ensemble_acc = val_ensemble_acc
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                best_ensemble_acc, coarse_mapping, Config.MODEL_SAVE_PATH
            )
            print(f'🎉 新的最佳模型! 准确率: {best_ensemble_acc:.4f}')

        # 定期保存检查点
        if epoch % 10 == 0:
            checkpoint_path = f'model/checkpoint_epoch_{epoch}.pth'
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                best_ensemble_acc, coarse_mapping, checkpoint_path
            )

    # 训练完成
    total_training_time = time.time() - train_start_time
    print(f'\n🎊 训练完成!')
    print(f'   总训练时间: {total_training_time / 3600:.1f} 小时')
    print(f'   最佳集成准确率: {best_ensemble_acc:.4f}')
    print(f'   最佳模型保存在: {Config.MODEL_SAVE_PATH}')

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f'✅ GPU内存已清理')


if __name__ == '__main__':
    main()
