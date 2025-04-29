import cv2
import numpy as np
import mss
import time
import ctypes
import os
import pynput
import winsound
from pynput.keyboard import Key, Listener
from ultralytics import YOLO

# === 初始化 Logitech 宏驱动 ===
try:
    driver = ctypes.CDLL('E:\\python_learn\\learn_pytorch\\pythonProject\\yolo_volant\\logi\\logitech.driver.dll')
    ok = driver.device_open() == 1
    if not ok:
        print('[ERROR] Logitech 驱动未连接')
except FileNotFoundError:
    print('[ERROR] Logitech DLL 未找到')
    ok = False

# === Logitech 控制类 ===
class Logitech:
    class mouse:
        @staticmethod
        def move(x, y):
            if not ok or (x == 0 and y == 0):
                return
            driver.moveR(int(x), int(y), True)
        @staticmethod
        def click(code=1):
            if not ok:
                return
            driver.mouse_down(code)
            driver.mouse_up(code)


# === 配置模型和参数 ===
MODEL_PATH = r"E:\python_learn\learn_pytorch\pythonProject\yolo_volant\save\0428_1\yolov8m_optimized_enemy\weights\best.pt"
CLASS_NAMES = ['enemy', 'enemy_head']
ROI_SIZE = 640
CONFIDENCE_THRESHOLD = 0.15
TARGET_CLASS = 'enemy_head'

# === 初始化模型（GPU） ===
model = YOLO(MODEL_PATH)
model.fuse()
print(f"[INFO] 使用模型设备: {model.device}")

# === 标志变量 ===
auto_aim_enabled = False  # 默认关闭自动瞄准
print("[INFO] 按 ` 开启/关闭自动瞄准，End 退出程序")

# === 热键控制 ===
def on_release(key):
    global auto_aim_enabled
    try:
        if key == Key.end:
            winsound.Beep(400, 200)
            print("[INFO] 脚本退出")
            return False
        elif key.char == '`':  # 检测 ` 键（ESC下方的符号）
            auto_aim_enabled = not auto_aim_enabled  # 切换自动瞄准状态
            winsound.Beep(600 if auto_aim_enabled else 300, 200)
            print(f"[INFO] 自动瞄准 {'开启' if auto_aim_enabled else '关闭'}")
    except AttributeError:
        pass

keyboard_listener = Listener(on_release=on_release)
keyboard_listener.start()

# === 启动主循环 ===
with mss.mss() as sct:
    screen = sct.monitors[1]
    screen_width = screen['width']
    screen_height = screen['height']
    left = screen['left'] + screen_width // 2 - ROI_SIZE // 2
    top = screen['top'] + screen_height // 2 - ROI_SIZE // 2
    monitor = {"top": top, "left": left, "width": ROI_SIZE, "height": ROI_SIZE}

    while True:
        start = time.time()
        # 获取屏幕截图
        frame = np.array(sct.grab(monitor))
        img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # 使用YOLO模型进行推理
        results = model.predict(
            img,
            conf=CONFIDENCE_THRESHOLD,
            classes=[CLASS_NAMES.index(TARGET_CLASS)],
            device=0,
            verbose=False
        )

        heads = []
        for r in results:
            for box in r.boxes.data:
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                heads.append((cx, cy))

        # 如果启用了自动瞄准，进行瞄准并开火
        if heads and auto_aim_enabled:
            nearest = min(heads, key=lambda p: (p[0] - ROI_SIZE / 2)**2 + (p[1] - ROI_SIZE / 2)**2)
            dx = nearest[0] - ROI_SIZE / 2
            dy = nearest[1] - ROI_SIZE / 2
            Logitech.mouse.move(dx, dy)

            # 在瞄准到目标后自动开火
            Logitech.mouse.click()  # 自动开火，模拟鼠标点击

        if not keyboard_listener.running:
            break

        # print(f"[INFO] 帧耗时: {(time.time() - start)*1000:.1f}ms")
