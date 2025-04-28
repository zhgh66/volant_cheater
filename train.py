from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
import ultralytics.utils.callbacks.tensorboard as tb_cb
import torch
import random


# 1. 创建 TensorBoard writer
writer = SummaryWriter("runs/yolov8_tensorboard")
tb_cb.WRITER = writer

# 全局变量跟踪当前epoch
current_epoch = 0

# 自定义：验证结束后打印并写入 TensorBoard
def log_pr_after_val(validator):
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

# def log_test_metrics(model, data_yaml):
#     """测试阶段输出并记录相关指标"""
#     try:
#         metrics = model.test(
#             data=data_yaml,
#             conf=0.25,  # 设置置信度阈值
#             iou_thres=0.3,  # 设置IoU阈值
#         )
#
#         # 提取指标并打印
#         precision = metrics.get('metrics/precision(B)', 0)
#         recall = metrics.get('metrics/recall(B)', 0)
#         f1 = 2 * precision * recall / (precision + recall + 1e-16)
#
#         print(f"Test Metrics - P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
#
#         # 写入 TensorBoard
#         writer.add_scalar("Test/P", precision)
#         writer.add_scalar("Test/R", recall)
#         writer.add_scalar("Test/F1", f1)
#
#     except Exception as e:
#         print(f"测试指标获取失败: {str(e)}")

def update_epoch_counter(trainer):
    """更新epoch计数器"""
    global current_epoch
    current_epoch = trainer.epoch


# 自定义：多尺度训练
def get_random_imgsz():
    """返回一个随机的训练图像大小，范围从 320 到 640"""
    return random.choice([320+320, 416+320, 512+320, 608+320, 1280])


# 主训练函数
def train_yolo(model_weights="yolov8l.pt",
               data_yaml="datasets/data.yaml",
               save_name="yolov8_tensorboard"):

    model = YOLO(model_weights)

    # 注册官方回调（TensorBoard 日志）
    model.add_callback("on_pretrain_routine_start", tb_cb.on_pretrain_routine_start)
    model.add_callback("on_train_epoch_end", tb_cb.on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", tb_cb.on_fit_epoch_end)

    # 注册自定义回调
    model.add_callback("on_val_end", log_pr_after_val)  # 验证结束回调
    # model.add_callback("on_test_end", log_test_metrics)  # 验证结束回调
    model.add_callback("on_train_epoch_end", update_epoch_counter)  # 更新epoch计数器



    # 开始训练
    model.train(
        data=data_yaml,
        epochs=100,
        imgsz=get_random_imgsz(),  # 使用多尺度训练
        batch=8,
        workers=0,
        name=save_name,
        patience=10,
        amp=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        cache=True,
        freeze=5 if hasattr(model.model, "backbone") else 0,

        optimizer="AdamW",
        lr0=1e-5,


    )

    # model.trainer.args.hyp.update({
    #     "box": 0.1,
    #     "cls": 0.5,
    #     "obj": 0.5,
    #     "iou": 1.0,
    #     "dfl": 0.1
    # })

    writer.close()

# 验证阶段
def visualize_errors(model_path, data_yaml):
    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        split='val',
        save=True,
        save_txt=True,
        save_crop=True,
        conf=0.25,
        # iou_thres=0.3
    )
    print("验证完成，错误图像已保存至 runs/detect/val/ 目录")


# 启动训练和验证
if __name__ == "__main__":

    train_yolo(
        model_weights="yolov8m.pt",
        data_yaml="datasets/data.yaml",
        save_name="yolov8m_optimized_enemy"
    )

    # model_path = "runs/detect/yolov8m_optimized_enemy/weights/best.pt"

    visualize_errors(
        model_path="runs/detect/yolov8m_optimized_enemy/weights/best.pt",
        data_yaml="datasets/data.yaml"
    )

    # model = YOLO(model_path)
    # log_test_metrics(model, data_yaml="datasets/data.yaml")  