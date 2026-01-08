import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import os
import sys

# 引入GUI库
try:
    import tkinter as tk
    from tkinter import messagebox, ttk # 使用ttk组件更美观
except ImportError:
    print("错误: 需要tkinter模块支持GUI。")
    sys.exit(1)

# ==================== 1. 模型定义 (保持不变) ====================
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==================== 2. 智能预处理 (核心算法) ====================
def smart_preprocessing(img_pil):
    """
    仿照MNIST数据集制作过程：
    1. 提取数字最小包围盒
    2. 缩放到20x20
    3. 居中放置在28x28的画布上
    """
    img = img_pil.convert('L')
    bbox = img.getbbox()
    
    if bbox is None:
        return img.resize((28, 28))
        
    digit = img.crop(bbox)
    w, h = digit.size
    
    # 保持长宽比缩放，最大边长为20
    max_side = max(w, h)
    scale = 20.0 / max_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    digit = digit.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 创建28x28黑底
    new_img = Image.new('L', (28, 28), 0)
    
    # 居中粘贴
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    new_img.paste(digit, (paste_x, paste_y))
    
    return new_img

# ==================== 3. 现代化 GUI 界面 ====================
class ModernApp:
    def __init__(self, root, model_path='lenet5_mnist.pth'):
        self.root = root
        self.root.title("LeNet-5 手写数字识别系统")
        self.root.geometry("700x520")
        
        # 尝试设置图标和样式
        style = ttk.Style()
        style.theme_use('clam')  # 使用更现代的主题
        
        # 定义颜色和字体
        self.bg_color = "#f5f6f7"
        self.root.configure(bg=self.bg_color)
        self.font_title = ("Microsoft YaHei", 16, "bold")
        self.font_normal = ("Microsoft YaHei", 10)
        self.font_digit = ("Arial", 60, "bold")

        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LeNet5().to(self.device)
        self.model_loaded = False
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.model_loaded = True
            except Exception as e:
                messagebox.showerror("错误", f"模型损坏: {e}")
        else:
            messagebox.showwarning("警告", f"找不到模型文件: {model_path}")

        # === 修复 Bug 的关键：加入 Pad(2) ===
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Pad(2)  # <--- 必须加这个！将28x28填充为32x32
        ])

        self._init_ui()

    def _init_ui(self):
        # 主容器
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # --- 左侧区域：绘图板 ---
        left_panel = tk.Frame(main_frame, bg=self.bg_color)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        tk.Label(left_panel, text="手写输入区", font=self.font_title, bg=self.bg_color, fg="#333").pack(anchor="w", pady=(0, 10))
        
        # 画布边框容器
        canvas_frame = tk.Frame(left_panel, bg="white", bd=2, relief="groove")
        canvas_frame.pack()
        
        self.canvas_size = 300
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size, 
                                bg="black", cursor="crosshair", highlightthickness=0)
        self.canvas.pack()
        
        # 提示文本
        tk.Label(left_panel, text="鼠标左键书写，右键也可清除", font=("Microsoft YaHei", 9), fg="#888", bg=self.bg_color).pack(pady=5)
        
        # 按钮组
        btn_frame = tk.Frame(left_panel, bg=self.bg_color)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="清除画布 (Clear)", command=self.clear_canvas).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # 绑定事件
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        self.canvas.bind("<Button-3>", lambda e: self.clear_canvas()) # 右键清除

        # --- 右侧区域：结果显示 ---
        right_panel = tk.Frame(main_frame, bg=self.bg_color)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tk.Label(right_panel, text="识别分析", font=self.font_title, bg=self.bg_color, fg="#333").pack(anchor="w", pady=(0, 10))
        
        # 结果卡片
        res_card = tk.Frame(right_panel, bg="white", bd=1, relief="solid")
        res_card.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(res_card, text="预测结果", bg="white", fg="#666", font=("Microsoft YaHei", 10)).pack(pady=(10, 0))
        self.lbl_pred = tk.Label(res_card, text="-", font=self.font_digit, bg="white", fg="#2196F3")
        self.lbl_pred.pack(pady=0)
        self.lbl_conf = tk.Label(res_card, text="等待输入...", bg="white", fg="#666", font=("Microsoft YaHei", 12))
        self.lbl_conf.pack(pady=(0, 15))

        # 概率图表
        tk.Label(right_panel, text="概率分布详情:", bg=self.bg_color, font=("Microsoft YaHei", 10, "bold")).pack(anchor="w")
        
        self.prob_canvas = tk.Canvas(right_panel, bg="white", height=200, highlightthickness=0)
        self.prob_canvas.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 识别按钮
        self.btn_predict = tk.Button(right_panel, text="立即识别 (Predict)", command=self.predict, 
                                     bg="#4CAF50", fg="white", font=("Microsoft YaHei", 12, "bold"), 
                                     relief="flat", cursor="hand2", pady=10)
        self.btn_predict.pack(fill=tk.X, side=tk.BOTTOM)

        # 内部绘图数据
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw_pil = ImageDraw.Draw(self.image)
        self.drawing = False
        self.last_point = None

    def start_draw(self, event):
        self.drawing = True
        self.last_point = (event.x, event.y)

    def draw(self, event):
        if self.drawing and self.last_point:
            x, y = event.x, event.y
            # 绘制线条 (更平滑)
            r = 8 # 笔触粗细
            self.canvas.create_line(self.last_point[0], self.last_point[1], x, y, 
                                    width=r*2, fill="white", capstyle=tk.ROUND, smooth=True)
            self.draw_pil.line([self.last_point, (x, y)], fill=255, width=r*2, joint="curve")
            self.last_point = (x, y)

    def stop_draw(self, event):
        self.drawing = False
        self.last_point = None
        # 自动识别体验更好，你可以注释掉下面这行改为手动点击
        # self.predict() 

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw_pil = ImageDraw.Draw(self.image)
        self.lbl_pred.config(text="-", fg="#2196F3")
        self.lbl_conf.config(text="等待输入...", fg="#666")
        self.prob_canvas.delete("all")

    def predict(self):
        if not self.model_loaded: return

        # 1. 检查是否为空
        if not self.canvas.find_all():
            self.lbl_conf.config(text="画布为空", fg="red")
            return

        # 2. 预处理 (转28x28居中)
        processed_img = smart_preprocessing(self.image)
        
        # 3. 转换为Tensor (这里会自动Pad到32x32)
        try:
            img_tensor = self.transform(processed_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
            
            digit = pred.item()
            confidence = conf.item()
            
            # 4. 更新UI
            self.lbl_pred.config(text=str(digit), fg="#e91e63") # 识别结果用醒目颜色
            self.lbl_conf.config(text=f"置信度: {confidence*100:.1f}%", fg="#333")
            self.draw_chart(probs.squeeze().cpu().numpy(), digit)
            
        except RuntimeError as e:
            messagebox.showerror("运行错误", f"模型输入尺寸不匹配。\n详情: {e}")

    def draw_chart(self, probs, target_digit):
        self.prob_canvas.delete("all")
        w = self.prob_canvas.winfo_width()
        h = self.prob_canvas.winfo_height()
        bar_h = h / 10
        
        for i, p in enumerate(probs):
            y0 = i * bar_h + 4
            y1 = (i + 1) * bar_h - 4
            
            # 绘制背景槽
            self.prob_canvas.create_rectangle(30, y0, w-50, y1, fill="#f0f0f0", outline="")
            
            # 绘制数据条
            bar_w = (w - 80) * p
            color = "#e91e63" if i == target_digit else "#90caf9" # 选中项红色，其他蓝色
            self.prob_canvas.create_rectangle(30, y0, 30 + bar_w, y1, fill=color, outline="")
            
            # 文字标签
            self.prob_canvas.create_text(15, (y0+y1)/2, text=str(i), fill="#555", font=("Arial", 10))
            self.prob_canvas.create_text(w-25, (y0+y1)/2, text=f"{p*100:.0f}%", fill="#888", font=("Arial", 8))

if __name__ == "__main__":
    root = tk.Tk()
    # 居中屏幕
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws/2) - (700/2)
    y = (hs/2) - (520/2)
    root.geometry('+%d+%d' % (x, y))
    
    app = ModernApp(root, model_path='lenet5_mnist.pth')
    root.mainloop()