# 梯度与 PyTorch 自动求导

> 整理时间：2026-05-13 | 课程：人机交互

---

## 一、梯度是什么

### 1.1 直观理解

梯度是**函数在某一点上变化最快的方向**。

想象你站在一座山上：
- 梯度指向**最陡的上坡方向**
- 梯度的长度（模长）表示**坡有多陡**
- 负梯度指向**最陡的下坡方向**（梯度下降的核心）

### 1.2 数学定义

对于多元函数 $f(x_1, x_2, \dots, x_n)$，梯度是该函数对所有自变量偏导数构成的向量：

$$\nabla f = \left( \frac{\partial f}{\partial x_1},\ \frac{\partial f}{\partial x_2},\ \dots,\ \frac{\partial f}{\partial x_n} \right)$$

### 1.3 一元函数特例

一元函数 $f(x)$ 的梯度退化为导数：

$$\nabla f(x) = f'(x)$$

### 1.4 为什么重要

- **优化问题**：梯度下降法是最基础的优化算法，沿负梯度方向迭代逼近极小值
- **深度学习**：反向传播的本质就是链式法则 + 梯度计算
- **物理意义**：势能函数的梯度等于力场（负号）

---

## 二、PyTorch 与梯度

### 2.1 PyTorch 中的梯度是什么

PyTorch 的 `autograd` 模块提供了**自动微分**（Automatic Differentiation）能力。核心机制：

- **计算图**：每次对 `requires_grad=True` 的张量进行运算时，PyTorch 会在后台构建一张有向无环图（DAG），记录所有运算步骤
- **反向传播**：调用 `.backward()` 时，沿计算图从输出反向遍历，用链式法则自动计算每个节点的梯度
- **梯度存储**：叶子节点的梯度累积在 `.grad` 属性中

### 2.2 关键概念

| 概念 | 说明 |
|------|------|
| `requires_grad=True` | 标记该张量需要跟踪梯度 |
| `.backward()` | 从当前张量开始反向计算梯度 |
| `.grad` | 叶子节点的梯度值（默认 None，反向传播后才有） |
| `torch.no_grad()` | 上下文管理器，禁用梯度跟踪（推理/评估时使用） |
| `.zero_grad()` | 清零梯度（每次反向传播前必须清零，否则梯度会累加） |

### 2.3 计算图的生命周期

```
前向传播：构建计算图 → 计算输出 → 得到标量 loss
反向传播：loss.backward() → 沿图反向 → 叶子节点 .grad 被填充
释放图：反向传播后计算图自动释放（除非设置 retain_graph=True）
```

### 2.4 为什么用 PyTorch 而不用手推

- 手推复杂函数的梯度容易出错，且工作量大
- 自动微分逐算子链式求导，精度为机器精度
- 可以处理任意复杂的计算图（含 if/while 控制流）

---

## 三、实验代码：手推梯度 + PyTorch 自动求导对比

### 3.1 实验设定

给定函数：

$$f(x, y) = x^3 + 2y^2 + 3xy$$

初始点：$(x_0, y_0) = (2.0, 3.0)$

### 3.2 手推梯度

**对 x 求偏导：**

$$\frac{\partial f}{\partial x} = \frac{\partial}{\partial x}(x^3 + 2y^2 + 3xy) = 3x^2 + 3y$$

**对 y 求偏导：**

$$\frac{\partial f}{\partial y} = \frac{\partial}{\partial y}(x^3 + 2y^2 + 3xy) = 4y + 3x$$

**在点 (2, 3) 处代入：**

$$\frac{\partial f}{\partial x}\bigg|_{(2,3)} = 3 \times 4 + 3 \times 3 = 12 + 9 = 21$$

$$\frac{\partial f}{\partial y}\bigg|_{(2,3)} = 4 \times 3 + 3 \times 2 = 12 + 6 = 18$$

**手推梯度：** $\nabla f(2,3) = (21, 18)$

### 3.3 PyTorch 代码实现

```python
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
```

### 3.4 运行输出

```
==================================================
【实验一】手推梯度 vs PyTorch 自动求导
==================================================

初始点: x = 2.0, y = 3.0
函数值: f(2, 3) = 44.0
(手算验证: 2^3 + 2*3^2 + 3*2*3 = 8 + 18 + 18 = 44)

PyTorch 自动求导结果:
  df/dx = 21.0
  df/dy = 18.0

手推公式结果:
  df/dx = 3*x^2 + 3*y = 3*4 + 9 = 21.0
  df/dy = 4*y + 3*x   = 12 + 6  = 18.0

两者一致: True
```

### 3.5 关键要点

| 步骤 | PyTorch 操作 | 说明 |
|------|-------------|------|
| 标记变量 | `requires_grad=True` | 告诉 autograd 跟踪此张量 |
| 前向计算 | 正常写算式 | 自动构建计算图 |
| 反向传播 | `.backward()` | 沿计算图反向求梯度 |
| 读取梯度 | `.grad` | 获取叶子节点的梯度值 |
| 清零梯度 | `.zero_grad()` | 多次迭代时必须清零 |

---

## 四、扩展：矩阵输入与梯度下降示例

当输入为矩阵、输出为标量时，梯度维度与输入一致：

```python
print("\n")
print("=" * 50)
print("【实验二】梯度下降 — 让机器自己学会一组参数")
print("=" * 50)

# 问题: 已知 X 和 y, 求一组权重来拟合它们
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
print(f"初始随机权重: {[f'{v:.3f}' for v in w.tolist()]}")

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
```

---

## 五、常见陷阱

1. **忘记 `zero_grad()`**：梯度默认累加，不归零会导致梯度越来越大
2. **非标量 backward**：对向量/矩阵调用 `.backward()` 需要传入 `gradient` 参数
3. **`with torch.no_grad()`**：推理/评估时必须用，否则计算图不断累积，内存泄漏
4. **in-place 操作**：对 `requires_grad=True` 的叶子张量做 in-place 修改会破坏计算图
