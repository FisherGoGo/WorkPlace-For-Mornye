import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch


print("=" * 50)
print("【实验一】手推梯度 vs PyTorch 自动求导")
print("=" * 50)

# 待分析函数: f(x, y) = x^3 + 2*y^2 + 3*x*y
# 在点 (x=2, y=3) 处求梯度

# --- 第一步: 创建变量 ---
# requires_grad=True 意思是"请帮我记录运算, 回头我要算梯度"
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

print(f"\n初始点: x = {x.item():.1f}, y = {y.item():.1f}")

# --- 第二步: 前向计算 ---
# 正常写算式就行, PyTorch 会在后台画一张"计算图"
z = x**3 + 2 * y**2 + 3 * x * y

print(f"函数值: f(2, 3) = {z.item():.1f}")
print(f"(手算验证: 2^3 + 2*3^2 + 3*2*3 = 8 + 18 + 18 = 44)")

# --- 第三步: 反向传播 ---
# .backward() 沿着计算图往回走, 自动算出每个变量的梯度
z.backward()

# --- 第四步: 查看结果 ---
# 梯度存在 .grad 属性里
dx_torch = x.grad.item()  # df/dx 在 (2,3) 处的值
dy_torch = y.grad.item()  # df/dy 在 (2,3) 处的值

print(f"\nPyTorch 自动求导结果:")
print(f"  df/dx = {dx_torch}")
print(f"  df/dy = {dy_torch}")

# --- 第五步: 和手推结果对比 ---
# 手推偏导公式: df/dx = 3x^2 + 3y,  df/dy = 4y + 3x
dx_manual = 3 * (2.0**2) + 3 * 3.0   # 3*4 + 9 = 21
dy_manual = 4 * 3.0 + 3 * 2.0        # 12 + 6 = 18

print(f"\n手推公式结果:")
print(f"  df/dx = 3*x^2 + 3*y = 3*4 + 9 = {dx_manual}")
print(f"  df/dy = 4*y + 3*x   = 12 + 6  = {dy_manual}")

print(f"\n两者一致: {dx_torch == dx_manual and dy_torch == dy_manual}")


print("\n")
print("=" * 50)
print("【实验二】梯度下降 — 让机器自己学会一组参数")
print("=" * 50)

# 问题: 已知 X 和 y, 求一条直线(其实是超平面)来拟合它们
# 真实关系: y = 2*x1 - 1.5*x2 + 3*x3 + 一点噪声

torch.manual_seed(42)  # 固定随机种子, 保证每次运行结果一样

# --- 第一步: 造数据 ---
N = 100  # 100 个样本
X = torch.randn(N, 3)                    # 随机输入, 形状 (100, 3)
true_w = torch.tensor([2.0, -1.5, 3.0])  # 真实权重(我们假装不知道)
y = X @ true_w + 0.1 * torch.randn(N)    # 真实标签 + 噪声

print(f"\n数据: X 形状 {X.shape}, y 形状 {y.shape}")
print(f"真实权重(隐藏): {true_w.tolist()}")

# --- 第二步: 初始化要训练的权重 ---
# 一开始随机猜, requires_grad=True 让 PyTorch 帮我们算梯度
w = torch.randn(3, requires_grad=True)
print(f"初始随机权重: {w.tolist()}")

# --- 第三步: 训练循环 ---
learning_rate = 0.1  # 每次更新的步长

for step in range(20):
    # 3.1 预测: y_pred = X @ w
    y_pred = X @ w

    # 3.2 算误差: MSE (均方误差)
    error = y_pred - y         # 每个样本的误差
    loss = (error ** 2).mean() # 误差平方的均值

    # 3.3 反向传播: 算 loss 对 w 的梯度
    loss.backward()

    # 3.4 更新 w: 沿负梯度方向走一小步
    # w = w - learning_rate * grad
    with torch.no_grad():
        w -= learning_rate * w.grad

    # 3.5 梯度清零 (非常重要! 不清零会累加)
    w.grad.zero_()

    # 每几轮打印一次
    if step < 5 or step >= 15:
        print(f"  第 {step+1:2d} 步: loss = {loss.item():.4f}")

print(f"\n训练好的权重: {[f'{v:.3f}' for v in w.tolist()]}")
print(f"真实权重:     {[f'{v:.3f}' for v in true_w.tolist()]}")
print(f"已经很接近了!")
