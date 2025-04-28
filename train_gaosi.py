import os
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
import ultralytics.utils.callbacks.tensorboard as tb_cb
import torch
import random
from pathlib import Path
import numpy as np
from PIL import Image



def add_gaussian_noise(image, mean=0, std=0.1):
    """给图像添加高斯噪声"""
    np_image = np.array(image)
    noise = np.random.normal(mean, std, np_image.shape)
    noisy_image = np_image + noise * 255  # 调整噪声尺度
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)


def gaussian_augment_train_data(train_data_path, augmented_data_path, noise_prob=0.1):
    """每轮训练前在训练集中随机选取10%的图像进行高斯噪声增强"""
    images_path = list(Path(train_data_path).glob('**/*.jpg'))  # 假设图像是jpg格式
    total_images = len(images_path)
    noisy_images_count = int(total_images * noise_prob)

    # 创建存储增强图像的目录
    os.makedirs(augmented_data_path, exist_ok=True)

    # 清空原来的增强图像文件夹（每次训练前刷新数据）
    for file in Path(augmented_data_path).glob('*'):
        file.unlink()

    # 随机选取需要增强的图像
    noisy_images = random.sample(images_path, noisy_images_count)

    for img_path in noisy_images:
        # 打开图片
        img = Image.open(img_path)
        img = add_gaussian_noise(img)

        # 创建一个新路径来保存增强后的图像
        new_img_path = Path(augmented_data_path) / img_path.name
        img.save(new_img_path)  # 保存增强后的图像
        print(f"Image {img_path} has been augmented with Gaussian noise and saved as {new_img_path}")


# 1. 创建 TensorBoard writer
writer = SummaryWriter("runs/yolov8_tensorboard")
tb_cb.WRITER = writer

# 全局变量跟踪当前epoch
current_epoch = 0


# 自定义：验证结束后打印并写入 TensorBoard
def log_pr_after_val(validator):
    """验证结束后记录指标的正确方式（最终修正版）"""
    global current_epoch

    try:
        # 获取验证指标（直接从结果字典获取）
        results = validator.metrics.results_dict

        # 提取指标（根据你的调试信息使用正确的键）
        precision = results.get("metrics/precision(B)", 0)
        recall = results.get("metrics/recall(B)", 0)

        # 计算 F1 值
        f1 = 2 * precision * recall / (precision + recall + 1e-16)

        # 打印到控制台（显示正确epoch）
        print(f"After Val - Epoch {current_epoch + 1}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

        # 写入 TensorBoard（使用正确epoch计数）
        writer.add_scalar("Val/P", precision, current_epoch + 1)
        writer.add_scalar("Val/R", recall, current_epoch + 1)
        writer.add_scalar("Val/F1", f1, current_epoch + 1)

    except Exception as e:
        print(f"指标获取失败: {str(e)}")


def update_epoch_counter(trainer):
    """更新epoch计数器"""
    global current_epoch
    current_epoch = trainer.epoch




# 主训练函数
def train_yolo(model_weights="yolov8l.pt",
               data_yaml="datasets/data.yaml",
               save_name="yolov8_tensorboard",
               augmented_data_path="datasets/train_augmented"):
    model = YOLO(model_weights)

    # 每轮训练前对训练集进行高斯噪声增强
    gaussian_augment_train_data('datasets/train', augmented_data_path, noise_prob=0.1)

    # 注册官方回调（TensorBoard 日志）
    model.add_callback("on_pretrain_routine_start", tb_cb.on_pretrain_routine_start)
    model.add_callback("on_train_epoch_end", tb_cb.on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", tb_cb.on_fit_epoch_end)

    # 注册自定义回调
    model.add_callback("on_val_end", log_pr_after_val)  # 验证结束回调
    model.add_callback("on_train_epoch_end", update_epoch_counter)  # 更新epoch计数器
    # model.add_callback("on_train_batch_start", log_batch_size)  # 记录输入尺寸

    # 开始训练（关键修改部分）
    model.train(
        data=data_yaml,
        epochs=200,
        imgsz=640,  # 基准尺寸
        batch=6,
        workers=0,
        name=save_name,
        patience=10,
        amp=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        cache=True,
        freeze=10 if hasattr(model.model, "backbone") else 0,
        optimizer="AdamW",
        lr0=1e-6,


        # 多尺度训练核心参数
        augment=True,
        multi_scale=True,
        scale=0.2,
        mosaic=1.0,
        mixup=0.1,
        erasing=0.2,
        fliplr=0.3,
        degrees=5.0,
        shear=1.0,
    )

    writer.close()


# 验证阶段（保持固定尺寸）
def visualize_errors(model_path, data_yaml):
    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        imgsz=640,  # 固定验证尺寸
        split='val',
        save=True,
        save_txt=True,
        save_crop=True,
        conf=0.25
    )
    print("验证完成，错误图像已保存至 runs/detect/val/ 目录")


# 启动训练和验证
if __name__ == "__main__":
    train_yolo(
        model_weights="yolov8m.pt",
        data_yaml="datasets/data.yaml",
        save_name="yolov8m_optimized_enemy"
    )

    visualize_errors(
        model_path="runs/detect/yolov8m_optimized_enemy/weights/best.pt",
        data_yaml="datasets/data.yaml"
    )