"""
=============================================================================
人工智能基础实验三 - LeNet-5识别手写数字
=============================================================================
本代码实现了经典的LeNet-5卷积神经网络模型，用于MNIST手写数字识别任务。
LeNet-5是由Yann LeCun教授在1998年提出的，是第一个成功应用于数字识别的CNN模型。

作者：学生
日期：2026年1月4日
=============================================================================
"""

# ==================== 导入必要的库 ====================
import torch                                    # PyTorch深度学习框架
import torch.nn as nn                           # 神经网络模块
import torch.nn.functional as F                 # 神经网络函数库
import torch.optim as optim                     # 优化器模块
from torch.utils.data import DataLoader         # 数据加载器
from torchvision import datasets, transforms   # 数据集和数据变换
import matplotlib.pyplot as plt                 # 绘图库
import numpy as np                              # 数值计算库
import time                                     # 时间模块
import os                                       # 操作系统模块


# ==================== LeNet-5网络模型定义 ====================
class LeNet5(nn.Module):
    """
    LeNet-5卷积神经网络模型
    
    网络结构：
    - 输入层: 32x32灰度图像（MNIST原图28x28，需要padding）
    - C1层: 卷积层，6个5x5卷积核，输出6@28x28
    - S2层: 池化层，2x2最大池化，输出6@14x14
    - C3层: 卷积层，16个5x5卷积核，输出16@10x10
    - S4层: 池化层，2x2最大池化，输出16@5x5
    - C5层: 卷积层，120个5x5卷积核，输出120@1x1
    - F6层: 全连接层，输出84个神经元
    - 输出层: 全连接层，输出10个类别
    """
    
    def __init__(self):
        """
        初始化LeNet-5网络的各个层
        """
        super(LeNet5, self).__init__()
        
        # ========== C1卷积层 ==========
        # 输入通道: 1（灰度图）
        # 输出通道: 6（6个特征图）
        # 卷积核大小: 5x5
        # 输出尺寸: (32-5)/1 + 1 = 28 -> 6@28x28
        self.conv1 = nn.Conv2d(
            in_channels=1,      # 输入通道数（灰度图像为1）
            out_channels=6,     # 输出通道数（6个卷积核）
            kernel_size=5,      # 卷积核大小5x5
            stride=1,           # 步长为1
            padding=0           # 无填充
        )
        
        # ========== S2池化层 ==========
        # 池化窗口: 2x2
        # 池化方式: 最大池化
        # 输出尺寸: 28/2 = 14 -> 6@14x14
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,      # 池化窗口大小2x2
            stride=2            # 步长为2
        )
        
        # ========== C3卷积层 ==========
        # 输入通道: 6
        # 输出通道: 16
        # 卷积核大小: 5x5
        # 输出尺寸: (14-5)/1 + 1 = 10 -> 16@10x10
        self.conv2 = nn.Conv2d(
            in_channels=6,      # 输入通道数
            out_channels=16,    # 输出通道数（16个卷积核）
            kernel_size=5,      # 卷积核大小5x5
            stride=1,           # 步长为1
            padding=0           # 无填充
        )
        
        # ========== S4池化层 ==========
        # 池化窗口: 2x2
        # 输出尺寸: 10/2 = 5 -> 16@5x5
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,      # 池化窗口大小2x2
            stride=2            # 步长为2
        )
        
        # ========== C5卷积层（可视为全连接） ==========
        # 输入: 16@5x5
        # 输出: 120@1x1
        # 由于输入尺寸正好是5x5，使用5x5卷积核，输出变为1x1
        self.conv3 = nn.Conv2d(
            in_channels=16,     # 输入通道数
            out_channels=120,   # 输出通道数（120个卷积核）
            kernel_size=5,      # 卷积核大小5x5
            stride=1,           # 步长为1
            padding=0           # 无填充
        )
        
        # ========== F6全连接层 ==========
        # 输入: 120个神经元
        # 输出: 84个神经元
        self.fc1 = nn.Linear(
            in_features=120,    # 输入特征数
            out_features=84     # 输出特征数
        )
        
        # ========== 输出层 ==========
        # 输入: 84个神经元
        # 输出: 10个类别（0-9数字）
        self.fc2 = nn.Linear(
            in_features=84,     # 输入特征数
            out_features=10     # 输出类别数（0-9）
        )
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入张量，形状为 (batch_size, 1, 32, 32)
            
        返回:
            输出张量，形状为 (batch_size, 10)
        """
        # C1卷积层 + 激活函数
        # 输入: (batch, 1, 32, 32) -> 输出: (batch, 6, 28, 28)
        x = F.relu(self.conv1(x))
        
        # S2池化层
        # 输入: (batch, 6, 28, 28) -> 输出: (batch, 6, 14, 14)
        x = self.pool1(x)
        
        # C3卷积层 + 激活函数
        # 输入: (batch, 6, 14, 14) -> 输出: (batch, 16, 10, 10)
        x = F.relu(self.conv2(x))
        
        # S4池化层
        # 输入: (batch, 16, 10, 10) -> 输出: (batch, 16, 5, 5)
        x = self.pool2(x)
        
        # C5卷积层 + 激活函数
        # 输入: (batch, 16, 5, 5) -> 输出: (batch, 120, 1, 1)
        x = F.relu(self.conv3(x))
        
        # 展平操作，将多维特征图转换为一维向量
        # 输入: (batch, 120, 1, 1) -> 输出: (batch, 120)
        x = x.view(-1, 120)
        
        # F6全连接层 + 激活函数
        # 输入: (batch, 120) -> 输出: (batch, 84)
        x = F.relu(self.fc1(x))
        
        # 输出层（不使用激活函数，后续使用CrossEntropyLoss会自动应用Softmax）
        # 输入: (batch, 84) -> 输出: (batch, 10)
        x = self.fc2(x)
        
        return x


# ==================== 数据预处理和加载 ====================
def load_data(batch_size=64):
    """
    加载和预处理MNIST数据集
    
    参数:
        batch_size: 批次大小，默认64
        
    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 定义数据预处理流程
    # 1. 将PIL图像或numpy数组转换为张量
    # 2. 将28x28图像填充到32x32（LeNet-5要求输入32x32）
    # 3. 标准化处理（均值0.1307，标准差0.3081是MNIST数据集的统计值）
    transform = transforms.Compose([
        transforms.ToTensor(),                          # 转换为张量，同时将像素值归一化到[0,1]
        transforms.Normalize((0.1307,), (0.3081,)),    # 标准化处理
        transforms.Pad(2)                               # 填充2个像素，28x28 -> 32x32
    ])
    
    # 创建数据存储目录
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 加载MNIST训练数据集
    # root: 数据存储路径
    # train: True表示加载训练集
    # download: True表示如果本地没有数据则自动下载
    # transform: 数据预处理变换
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # 加载MNIST测试数据集
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    # batch_size: 每个批次的样本数
    # shuffle: 是否打乱数据顺序（训练时打乱，测试时不打乱）
    # num_workers: 数据加载的并行进程数
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # 训练时打乱数据
        num_workers=0           # Windows下建议设为0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # 测试时不打乱数据
        num_workers=0
    )
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"批次大小: {batch_size}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    return train_loader, test_loader


# ==================== 训练函数 ====================
def train(model, device, train_loader, optimizer, criterion, epoch):
    """
    训练模型一个epoch
    
    参数:
        model: 神经网络模型
        device: 计算设备（CPU或GPU）
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        epoch: 当前epoch编号
        
    返回:
        avg_loss: 平均损失值
        accuracy: 训练准确率
    """
    model.train()  # 设置模型为训练模式
    
    running_loss = 0.0      # 累计损失
    correct = 0             # 正确预测数
    total = 0               # 总样本数
    
    # 遍历训练数据
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据移动到计算设备
        data, target = data.to(device), target.to(device)
        
        # 清零梯度（每个batch开始时需要清零上一次的梯度）
        optimizer.zero_grad()
        
        # 前向传播：计算预测值
        output = model(data)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播：计算梯度
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)  # 获取预测类别
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 每100个batch打印一次训练状态
        if (batch_idx + 1) % 100 == 0:
            print(f'  Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}')
    
    # 计算平均损失和准确率
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


# ==================== 测试函数 ====================
def test(model, device, test_loader, criterion):
    """
    在测试集上评估模型
    
    参数:
        model: 神经网络模型
        device: 计算设备
        test_loader: 测试数据加载器
        criterion: 损失函数
        
    返回:
        avg_loss: 平均损失值
        accuracy: 测试准确率
    """
    model.eval()  # 设置模型为评估模式
    
    test_loss = 0.0     # 累计损失
    correct = 0         # 正确预测数
    total = 0           # 总样本数
    
    # 不计算梯度（节省内存和计算资源）
    with torch.no_grad():
        for data, target in test_loader:
            # 将数据移动到计算设备
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 累计损失
            test_loss += criterion(output, target).item()
            
            # 统计正确预测数
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # 计算平均损失和准确率
    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


# ==================== 可视化函数 ====================
def visualize_samples(test_loader, model, device, num_samples=10):
    """
    可视化部分测试样本及其预测结果
    
    参数:
        test_loader: 测试数据加载器
        model: 训练好的模型
        device: 计算设备
        num_samples: 显示的样本数量
    """
    model.eval()
    
    # 获取一个batch的数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # 预测
    with torch.no_grad():
        images_device = images.to(device)
        outputs = model(images_device)
        _, predictions = torch.max(outputs, 1)
    
    # 创建图形
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('LeNet-5 MNIST手写数字识别结果', fontsize=14)
    
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            # 显示图像（去掉padding，显示原始28x28区域）
            img = images[i].squeeze().numpy()[2:-2, 2:-2]  # 去掉2像素的padding
            ax.imshow(img, cmap='gray')
            
            # 设置标题（真实标签 vs 预测标签）
            true_label = labels[i].item()
            pred_label = predictions[i].item()
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f'真实: {true_label}\n预测: {pred_label}', 
                        color=color, fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("预测结果图已保存为 prediction_results.png")


def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """
    绘制训练历史曲线
    
    参数:
        train_losses: 训练损失列表
        train_accs: 训练准确率列表
        test_losses: 测试损失列表
        test_accs: 测试准确率列表
    """
    epochs = range(1, len(train_losses) + 1)
    
    # 创建包含两个子图的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 绘制损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='测试损失', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('训练过程中的损失变化', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='训练准确率', linewidth=2)
    ax2.plot(epochs, test_accs, 'r-', label='测试准确率', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('训练过程中的准确率变化', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("训练历史图已保存为 training_history.png")


def visualize_feature_maps(model, test_loader, device):
    """
    可视化卷积层的特征图
    
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
    """
    model.eval()
    
    # 获取一个样本
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    image = images[0:1].to(device)  # 取第一个样本
    
    # 提取各层特征图
    with torch.no_grad():
        # C1层特征图
        c1_output = F.relu(model.conv1(image))
        # S2层特征图
        s2_output = model.pool1(c1_output)
        # C3层特征图
        c3_output = F.relu(model.conv2(s2_output))
    
    # 可视化C1层的6个特征图
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(f'C1卷积层特征图 (输入数字: {labels[0].item()})', fontsize=14)
    
    # 显示原始图像
    axes[0, 0].imshow(images[0].squeeze().numpy()[2:-2, 2:-2], cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 显示C1层的6个特征图
    c1_maps = c1_output[0].cpu().numpy()
    for i in range(6):
        row = (i + 1) // 4
        col = (i + 1) % 4
        axes[row, col].imshow(c1_maps[i], cmap='viridis')
        axes[row, col].set_title(f'特征图 {i+1}')
        axes[row, col].axis('off')
    
    axes[1, 3].axis('off')  # 隐藏多余的子图
    
    plt.tight_layout()
    plt.savefig('feature_maps.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("特征图已保存为 feature_maps.png")


# ==================== 主函数 ====================
def main():
    """
    主函数：完整的训练和测试流程
    """
    print("=" * 60)
    print("LeNet-5 手写数字识别系统")
    print("=" * 60)
    
    # ========== 1. 设置超参数 ==========
    BATCH_SIZE = 64         # 批次大小
    EPOCHS = 10             # 训练轮数
    LEARNING_RATE = 0.001   # 学习率
    
    print(f"\n超参数设置:")
    print(f"  批次大小 (Batch Size): {BATCH_SIZE}")
    print(f"  训练轮数 (Epochs): {EPOCHS}")
    print(f"  学习率 (Learning Rate): {LEARNING_RATE}")
    
    # ========== 2. 设置计算设备 ==========
    # 如果有GPU则使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n计算设备: {device}")
    
    # ========== 3. 加载数据 ==========
    print("\n" + "-" * 40)
    print("加载MNIST数据集...")
    train_loader, test_loader = load_data(BATCH_SIZE)
    
    # ========== 4. 创建模型 ==========
    print("\n" + "-" * 40)
    print("创建LeNet-5模型...")
    model = LeNet5().to(device)
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数总数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # ========== 5. 定义损失函数和优化器 ==========
    # 使用交叉熵损失函数（适用于多分类问题）
    criterion = nn.CrossEntropyLoss()
    
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n损失函数: CrossEntropyLoss")
    print(f"优化器: Adam (lr={LEARNING_RATE})")
    
    # ========== 6. 训练模型 ==========
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    # 记录训练历史
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 40)
        
        # 训练一个epoch
        train_loss, train_acc = train(model, device, train_loader, 
                                       optimizer, criterion, epoch)
        
        # 在测试集上评估
        test_loss, test_acc = test(model, device, test_loader, criterion)
        
        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 打印epoch结果
        print(f"\n  训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
        print(f"  测试集 - 损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # ========== 7. 输出最终结果 ==========
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"总训练时间: {training_time:.2f} 秒")
    print(f"最终测试准确率: {test_accs[-1]:.2f}%")
    
    # ========== 8. 保存模型 ==========
    model_path = 'lenet5_mnist.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")
    
    # ========== 9. 可视化结果 ==========
    print("\n" + "-" * 40)
    print("生成可视化图表...")
    
    # 绘制训练历史曲线
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    
    # 可视化预测结果
    visualize_samples(test_loader, model, device)
    
    # 可视化特征图
    visualize_feature_maps(model, test_loader, device)
    
    print("\n程序运行完成!")
    
    return model, test_accs[-1]


# ==================== 程序入口 ====================
if __name__ == "__main__":
    # 设置matplotlib支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行主程序
    model, accuracy = main()
