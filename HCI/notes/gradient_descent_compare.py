import sys
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
# 如果中文显示有问题, 可以把下面这行注释掉
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']

import numpy as np
import torch


# ============================================================
# 在这里改函数定义, 两个版本会自动适配
# 默认: f(x, y) = x^2 + y^2   (碗形, 最低点在原点)
# ============================================================

def f_numpy(x, y):
    """任意二元函数 — 纯 Python / NumPy 版"""
    return x**2 + y**2
    # return x**4 + y**4                       # 试试这个: 更陡的碗
    # return x**2 + 10*y**2                    # 试试这个: 拉长的碗
    # return (x-3)**2 + (y+2)**2               # 试试这个: 最低点不在原点

def f_torch(x, y):  
    """任意二元函数 — PyTorch 版"""
    return x**2 + y**2
    # return x**4 + y**4
    # return x**2 + 10*y**2
    # return (x-3)**2 + (y+2)**2


# ============================================================
# 版本一: 纯 Python + 数值微分
# ============================================================

def numerical_gradient(f, x, y, h=1e-5):
    """用数值微分近似梯度: df/dx ≈ (f(x+h) - f(x-h)) / 2h"""
    dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return dx, dy


def gradient_descent_pure_python(f, init_x, init_y, lr=0.1, steps=50):
    """纯 Python 梯度下降, 返回每一步的记录"""
    x, y = init_x, init_y
    history = []

    for i in range(steps):
        loss = f(x, y)
        dx, dy = numerical_gradient(f, x, y)
        x -= lr * dx
        y -= lr * dy
        history.append((i, loss, x, y))

    return history


# ============================================================
# 版本二: PyTorch 自动求导
# ============================================================

def gradient_descent_pytorch(f, init_x, init_y, lr=0.1, steps=50):
    """PyTorch 自动求导梯度下降, 返回每一步的记录"""
    x = torch.tensor(init_x, requires_grad=True)
    y = torch.tensor(init_y, requires_grad=True)
    history = []

    for i in range(steps):
        loss = f(x, y)
        loss.backward()

        with torch.no_grad():
            x -= lr * x.grad
            y -= lr * y.grad

        x.grad.zero_()
        y.grad.zero_()

        history.append((i, loss.item(), x.item(), y.item()))

    return history


# ============================================================
# 绘图
# ============================================================

def plot_results(py_history, torch_history, init_xy, func_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- 左图: Loss 下降曲线 ----
    ax = axes[0]
    py_steps = [h[0] for h in py_history]
    py_loss = [h[1] for h in py_history]
    torch_steps = [h[0] for h in torch_history]
    torch_loss = [h[1] for h in torch_history]

    ax.plot(py_steps, py_loss, 'o-', markersize=3, linewidth=1.5,
            label='Pure Python (数值微分)', color='#3498db')
    ax.plot(torch_steps, torch_loss, 's-', markersize=3, linewidth=1.5,
            label='PyTorch (自动求导)', color='#e74c3c')
    ax.set_xlabel('迭代步数')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss 下降曲线\n{func_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 右图: 参数更新轨迹 (等高线 + 箭头) ----
    ax = axes[1]

    # 画等高线
    xs = np.linspace(min(-5, init_xy[0]) - 1, max(5, init_xy[0]) + 1, 100)
    ys = np.linspace(min(-5, init_xy[1]) - 1, max(5, init_xy[1]) + 1, 100)
    X, Y = np.meshgrid(xs, ys)
    Z = f_numpy(X, Y)

    levels = 20
    ax.contour(X, Y, Z, levels=levels, cmap='Blues', alpha=0.5)

    # PyTorch 轨迹
    tx = [h[2] for h in torch_history]
    ty = [h[3] for h in torch_history]
    ax.plot(tx, ty, 'o-', markersize=3, linewidth=1.5,
            color='#e74c3c', label='PyTorch 路径')
    ax.scatter([tx[0]], [ty[0]], color='orange', s=80, zorder=5, label='起点')
    ax.scatter([tx[-1]], [ty[-1]], color='green', s=80, zorder=5, label='终点')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'参数更新轨迹\n起点: ({init_xy[0]:.1f}, {init_xy[1]:.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gradient_descent_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图片已保存至: gradient_descent_plot.png")


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    # 随机初始值
    init_x = float(np.random.uniform(-5, 5))
    init_y = float(np.random.uniform(-5, 5))
    lr = 0.1
    steps = 50

    print("=" * 55)
    print("梯度下降 — 纯 Python vs PyTorch")
    print("=" * 55)
    print(f"\n函数: f(x, y) = {f_numpy.__doc__ or '见代码定义'}")
    print(f"初始点: ({init_x:.4f}, {init_y:.4f})")
    print(f"学习率: {lr},  迭代: {steps} 步\n")

    # 纯 Python 版
    print("--- 纯 Python (数值微分) ---")
    py_hist = gradient_descent_pure_python(f_numpy, init_x, init_y, lr, steps)
    print(f"  起点: f = {py_hist[0][1]:.6f}")
    print(f"  终点: f = {py_hist[-1][1]:.6f},  (x, y) = ({py_hist[-1][2]:.6f}, {py_hist[-1][3]:.6f})")

    # PyTorch 版
    print("\n--- PyTorch (自动求导) ---")
    torch_hist = gradient_descent_pytorch(f_torch, init_x, init_y, lr, steps)
    print(f"  起点: f = {torch_hist[0][1]:.6f}")
    print(f"  终点: f = {torch_hist[-1][1]:.6f},  (x, y) = ({torch_hist[-1][2]:.6f}, {torch_hist[-1][3]:.6f})")

    # 绘图
    plot_results(py_hist, torch_hist, (init_x, init_y),
                 func_name='f(x,y) = x^2 + y^2')
