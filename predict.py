"""
=============================================================================
LeNet-5 手写数字识别应用程序
=============================================================================
本程序用于加载训练好的LeNet-5模型，并应用于手写数字识别任务。

功能：
1. 从图片文件识别手写数字
2. 交互式手写画板识别
3. 批量识别测试

作者：学生
日期：2026年1月6日
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os

# ==================== LeNet-5模型定义（与训练时相同） ====================
class LeNet5(nn.Module):
    """
    LeNet-5卷积神经网络模型
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)
    
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


# ==================== 模型加载类 ====================
class DigitRecognizer:
    """
    手写数字识别器
    
    用于加载训练好的LeNet-5模型并进行预测
    """
    
    def __init__(self, model_path='lenet5_mnist.pth'):
        """
        初始化识别器
        
        参数:
            model_path: 模型权重文件路径
        """
        # 设置计算设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 创建模型实例
        self.model = LeNet5().to(self.device)
        
        # 加载训练好的权重
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"成功加载模型: {model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}\n请先运行 lenet5_mnist.py 训练模型")
        
        # 设置为评估模式
        self.model.eval()
        
        # 定义图像预处理流程
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
            transforms.Resize((28, 28)),                   # 调整大小为28x28
            transforms.ToTensor(),                         # 转换为张量
            transforms.Normalize((0.1307,), (0.3081,)),   # 标准化
            transforms.Pad(2)                              # 填充到32x32
        ])
    
    def preprocess_image(self, image):
        """
        预处理图像
        
        参数:
            image: PIL Image对象或图像文件路径
            
        返回:
            预处理后的张量
        """
        # 如果是路径，加载图像
        if isinstance(image, str):
            image = Image.open(image)
        
        # 确保是PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # 转换为灰度图
        if image.mode != 'L':
            image = image.convert('L')
        
        # 如果背景是白色，数字是黑色，需要反转
        # MNIST数据集是黑底白字
        image_array = np.array(image)
        if np.mean(image_array) > 127:  # 如果平均值大于127，说明是白底
            image = ImageOps.invert(image)
        
        # 应用预处理变换
        tensor = self.transform(image)
        
        # 添加batch维度
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict(self, image):
        """
        预测手写数字
        
        参数:
            image: PIL Image对象、图像文件路径或numpy数组
            
        返回:
            predicted_digit: 预测的数字 (0-9)
            confidence: 置信度 (0-1)
            probabilities: 各类别的概率分布
        """
        # 预处理图像
        tensor = self.preprocess_image(image)
        tensor = tensor.to(self.device)
        
        # 进行预测
        with torch.no_grad():
            output = self.model(tensor)
            
            # 使用softmax获取概率分布
            probabilities = F.softmax(output, dim=1)
            
            # 获取预测类别和置信度
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_digit = predicted.item()
            confidence = confidence.item()
            probabilities = probabilities.squeeze().cpu().numpy()
        
        return predicted_digit, confidence, probabilities
    
    def predict_from_file(self, image_path):
        """
        从图像文件预测数字
        
        参数:
            image_path: 图像文件路径
        """
        if not os.path.exists(image_path):
            print(f"错误: 文件不存在 - {image_path}")
            return
        
        # 预测
        digit, confidence, probs = self.predict(image_path)
        
        # 显示结果
        print(f"\n{'='*50}")
        print(f"图像文件: {image_path}")
        print(f"{'='*50}")
        print(f"预测结果: {digit}")
        print(f"置信度: {confidence*100:.2f}%")
        print(f"\n各数字的概率:")
        for i, prob in enumerate(probs):
            bar = '█' * int(prob * 30)
            print(f"  {i}: {bar} {prob*100:.2f}%")
        
        # 可视化
        self._visualize_prediction(image_path, digit, confidence, probs)
        
        return digit, confidence
    
    def _visualize_prediction(self, image_path, digit, confidence, probs):
        """
        可视化预测结果
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 显示原始图像
        img = Image.open(image_path)
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'输入图像\n预测: {digit} (置信度: {confidence*100:.1f}%)', fontsize=14)
        axes[0].axis('off')
        
        # 显示概率分布
        colors = ['green' if i == digit else 'steelblue' for i in range(10)]
        axes[1].barh(range(10), probs, color=colors)
        axes[1].set_yticks(range(10))
        axes[1].set_yticklabels([str(i) for i in range(10)])
        axes[1].set_xlabel('概率', fontsize=12)
        axes[1].set_title('各数字的预测概率', fontsize=14)
        axes[1].set_xlim(0, 1)
        
        # 添加概率值标签
        for i, prob in enumerate(probs):
            axes[1].text(prob + 0.02, i, f'{prob*100:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('prediction_output.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\n结果已保存为 prediction_output.png")


# ==================== 交互式手写画板 ====================
class HandwritingCanvas:
    """
    交互式手写画板
    
    使用matplotlib创建一个简单的画板，用户可以用鼠标手写数字
    """
    
    def __init__(self, recognizer):
        """
        初始化画板
        
        参数:
            recognizer: DigitRecognizer实例
        """
        self.recognizer = recognizer
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # 创建画布 (200x200像素)
        self.canvas_size = 200
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), color=0)  # 黑色背景
        self.draw = ImageDraw.Draw(self.image)
        
    def start(self):
        """
        启动交互式画板
        """
        print("\n" + "="*50)
        print("交互式手写数字识别")
        print("="*50)
        print("操作说明:")
        print("  - 按住鼠标左键绘制数字")
        print("  - 按 'r' 键清除画布")
        print("  - 按 'p' 键进行预测")
        print("  - 按 'q' 键或关闭窗口退出")
        print("="*50 + "\n")
        
        # 创建图形窗口
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(0, self.canvas_size)
        self.ax.set_ylim(self.canvas_size, 0)  # 翻转y轴
        self.ax.set_aspect('equal')
        self.ax.set_title('手写画板 (按r清除, 按p预测, 按q退出)', fontsize=12)
        self.ax.axis('off')
        
        # 显示画布
        self.img_display = self.ax.imshow(self.image, cmap='gray', vmin=0, vmax=255)
        
        # 连接事件
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        plt.show()
    
    def _on_press(self, event):
        """鼠标按下事件"""
        if event.inaxes == self.ax:
            self.drawing = True
            self.last_x = event.xdata
            self.last_y = event.ydata
    
    def _on_release(self, event):
        """鼠标释放事件"""
        self.drawing = False
        self.last_x = None
        self.last_y = None
    
    def _on_move(self, event):
        """鼠标移动事件"""
        if self.drawing and event.inaxes == self.ax:
            if self.last_x is not None and self.last_y is not None:
                # 绘制线条
                self.draw.line(
                    [(self.last_x, self.last_y), (event.xdata, event.ydata)],
                    fill=255,  # 白色
                    width=5   # 线条粗细
                )
                self.last_x = event.xdata
                self.last_y = event.ydata
                
                # 更新显示
                self.img_display.set_data(self.image)
                self.fig.canvas.draw_idle()
    
    def _on_key(self, event):
        """键盘事件"""
        if event.key == 'r':
            # 清除画布
            self.image = Image.new('L', (self.canvas_size, self.canvas_size), color=0)
            self.draw = ImageDraw.Draw(self.image)
            self.img_display.set_data(self.image)
            self.fig.canvas.draw_idle()
            print("画布已清除")
            
        elif event.key == 'p':
            # 进行预测
            self._predict()
            
        elif event.key == 'q':
            # 退出
            plt.close(self.fig)
    
    def _predict(self):
        """对当前画布内容进行预测"""
        # 检查画布是否为空
        img_array = np.array(self.image)
        if np.max(img_array) == 0:
            print("画布为空，请先绘制数字！")
            return
        
        # 预测
        digit, confidence, probs = self.recognizer.predict(self.image)
        
        # 显示结果
        print(f"\n预测结果: {digit} (置信度: {confidence*100:.2f}%)")
        
        # 更新标题
        self.ax.set_title(f'预测结果: {digit} (置信度: {confidence*100:.1f}%)', fontsize=14)
        self.fig.canvas.draw_idle()


# ==================== 批量测试函数 ====================
def batch_test(recognizer, image_folder):
    """
    批量测试图像文件夹中的所有图像
    
    参数:
        recognizer: DigitRecognizer实例
        image_folder: 图像文件夹路径
    """
    if not os.path.exists(image_folder):
        print(f"错误: 文件夹不存在 - {image_folder}")
        return
    
    # 支持的图像格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print(f"文件夹中没有找到图像文件: {image_folder}")
        return
    
    print(f"\n找到 {len(image_files)} 个图像文件")
    print("="*60)
    
    results = []
    for filename in image_files:
        filepath = os.path.join(image_folder, filename)
        try:
            digit, confidence, _ = recognizer.predict(filepath)
            results.append((filename, digit, confidence))
            print(f"{filename}: 预测={digit}, 置信度={confidence*100:.2f}%")
        except Exception as e:
            print(f"{filename}: 处理失败 - {e}")
    
    print("="*60)
    print(f"处理完成，共 {len(results)} 个文件")
    
    return results


# ==================== 主程序 ====================
def main():
    """
    主程序入口
    """
    print("="*60)
    print("LeNet-5 手写数字识别应用")
    print("="*60)
    
    # 加载模型
    try:
        recognizer = DigitRecognizer('lenet5_mnist.pth')
    except FileNotFoundError as e:
        print(e)
        return
    
    while True:
        print("\n请选择功能:")
        print("  1. 从图像文件识别")
        print("  2. 交互式手写画板")
        print("  3. 批量识别文件夹")
        print("  4. 测试MNIST样本")
        print("  0. 退出")
        
        choice = input("\n请输入选项 (0-4): ").strip()
        
        if choice == '1':
            # 从图像文件识别
            image_path = input("请输入图像文件路径: ").strip()
            if image_path:
                recognizer.predict_from_file(image_path)
            
        elif choice == '2':
            # 交互式手写画板
            canvas = HandwritingCanvas(recognizer)
            canvas.start()
            
        elif choice == '3':
            # 批量识别
            folder_path = input("请输入图像文件夹路径: ").strip()
            if folder_path:
                batch_test(recognizer, folder_path)
            
        elif choice == '4':
            # 测试MNIST样本
            test_mnist_samples(recognizer)
            
        elif choice == '0':
            print("感谢使用，再见！")
            break
        
        else:
            print("无效选项，请重新输入")


def test_mnist_samples(recognizer):
    """
    使用MNIST测试样本进行测试
    """
    try:
        from torchvision import datasets, transforms
        
        # 加载MNIST测试集
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        
        print("\n随机选择10个MNIST测试样本进行预测:")
        print("="*50)
        
        # 随机选择10个样本
        indices = np.random.choice(len(test_dataset), 10, replace=False)
        
        correct = 0
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        
        for idx, ax in zip(indices, axes.flat):
            image, true_label = test_dataset[idx]
            
            # 转换为PIL Image进行预测
            pil_image = transforms.ToPILImage()(image)
            digit, confidence, _ = recognizer.predict(pil_image)
            
            # 检查是否正确
            is_correct = (digit == true_label)
            if is_correct:
                correct += 1
            
            # 显示图像
            ax.imshow(image.squeeze().numpy(), cmap='gray')
            color = 'green' if is_correct else 'red'
            ax.set_title(f'真实: {true_label}, 预测: {digit}\n置信度: {confidence*100:.1f}%', 
                        color=color, fontsize=10)
            ax.axis('off')
        
        plt.suptitle(f'MNIST测试样本预测结果 (正确率: {correct}/10)', fontsize=14)
        plt.tight_layout()
        plt.savefig('mnist_test_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n预测正确: {correct}/10")
        print("结果已保存为 mnist_test_results.png")
        
    except Exception as e:
        print(f"测试失败: {e}")


# ==================== 程序入口 ====================
if __name__ == "__main__":
    # 设置matplotlib支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    main()
