import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import torch
import yolov5


class FlowerRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("智能花卉识别系统")
        self.master.geometry("1000x700")

        # 初始化YOLOv5模型
        self.model = yolov5.load('./best.pt')
        self.model.conf = 0.25  # 置信度阈值
        self.model.iou = 0.45  # NMS IoU阈值

        # 设置主题和样式
        self.style = ttk.Style(theme='minty')
        self.style.configure('TButton', font=('微软雅黑', 12))
        self.style.configure('TLabel', font=('微软雅黑', 12))

        # 创建界面布局
        self.create_widgets()

    def create_widgets(self):
        # 顶部标题
        header = ttk.Frame(self.master)
        header.pack(pady=20)
        ttk.Label(header, text="🌸 智能花卉识别系统", font=('微软雅黑', 24, 'bold')).pack()

        # 主体框架
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="操作面板", bootstyle=INFO)
        control_frame.pack(side=LEFT, fill=Y, padx=10, pady=10)

        # 上传按钮
        upload_btn = ttk.Button(
            control_frame,
            text="上传图片",
            command=self.upload_image,
            bootstyle=(OUTLINE, INFO),
            width=15
        )
        upload_btn.pack(pady=15)

        # 结果展示
        self.result_frame = ttk.LabelFrame(control_frame, text="识别结果", bootstyle=INFO)
        self.result_frame.pack(fill=X, pady=10)

        # 右侧图片显示
        self.image_frame = ttk.LabelFrame(main_frame, text="图片预览", bootstyle=INFO)
        self.image_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)

        # 增加Canvas的大小，fill=BOTH会让它占据父容器的空间
        self.canvas = tk.Canvas(self.image_frame, bg='#f8f9fa', bd=0, highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)  # 通过expand=True让canvas占据更多空间

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.process_image(file_path)

    def process_image(self, image_path):
        try:
            # 执行推理
            results = self.model(image_path)
            predictions = results.pred[0]
            classes = results.names

            # 加载图片
            img = cv2.imread(image_path)

            # 解析结果
            detected_objects = []
            for *box, conf, cls in predictions:
                class_name = classes[int(cls)]
                confidence = float(conf)
                detected_objects.append({
                    "class": class_name,
                    "confidence": f"{confidence:.2%}",
                    "position": [f"{x:.0f}" for x in box]
                })

                # 手动绘制矩形框
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 增加矩形框的宽度

                # 绘制文本并确保不超出图片边界
                label = f"{class_name} {confidence:.2f}"
                font_scale = 0.7  # 增大字体大小
                thickness = 1  # 增加文本线条的粗细
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

                # 确保文本不超出图像边界
                text_x = x1
                text_y = y1 - 10
                if text_y < 10:
                    text_y = y1 + 10

                # 添加抗锯齿效果
                cv2.putText(img, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, lineType=cv2.LINE_AA)

            # 显示结果
            self.show_results(detected_objects)

            # 转换为RGB图像后显示
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.show_image(img)

        except Exception as e:
            print(str(e))
            self.show_error(f"处理图片时出错: {str(e)}")

    def show_results(self, results):
        # 清空之前的结果
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        if not results:
            ttk.Label(self.result_frame, text="未检测到花卉", foreground="gray").pack()
            return

        # 创建表格展示结果
        columns = ['花卉类型', '置信度', '位置']
        tree = ttk.Treeview(
            self.result_frame,
            columns=columns,
            show='headings',
            height=min(len(results), 5),
            bootstyle=INFO
        )

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor=CENTER)

        for obj in results:
            tree.insert('', END, values=(
                obj['class'],
                obj['confidence'],
                f"({obj['position'][0]}, {obj['position'][1]})"
            ))

        tree.pack(fill=X)

    def show_image(self, image_array):
        # 转换图片格式
        img = Image.fromarray(image_array)

        # 获取Canvas的宽度和高度
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # 获取图像的原始宽度和高度
        img_width, img_height = img.size

        # 计算缩放比例，确保图片宽高比不变，适应Canvas的大小
        scale_width = canvas_width / img_width
        scale_height = canvas_height / img_height
        scale = min(scale_width, scale_height)  # 保证宽高比例适应Canvas

        # 计算调整后的图像尺寸
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        # 调整图像大小
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)  # 使用LANCZOS代替ANTIALIAS

        # 更新Canvas显示
        self.canvas.delete("all")
        self.tk_image = ImageTk.PhotoImage(img)

        # 居中显示图片
        x = (canvas_width - self.tk_image.width()) // 2
        y = (canvas_height - self.tk_image.height()) // 2
        self.canvas.create_image(x, y, anchor=NW, image=self.tk_image)

    def show_error(self, message):
        error_window = ttk.Toplevel(self.master)
        error_window.title("错误提示")

        ttk.Label(error_window, text=message, foreground="red", padding=10).pack()
        ttk.Button(
            error_window,
            text="确定",
            command=error_window.destroy,
            bootstyle=DANGER
        ).pack(pady=10)


if __name__ == "__main__":
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print("检测到CUDA加速可用")
    else:
        print("警告：未检测到CUDA，将使用CPU运行")

    # 创建界面
    root = ttk.Window()
    app = FlowerRecognitionApp(root)
    root.mainloop()