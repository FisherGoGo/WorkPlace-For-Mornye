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

# ========== 1. 定义函数 ==========
def f(x, y):
    """待求梯度的函数：f(x,y) = x^3 + 2*y^2 + 3*x*y"""
    return x**3 + 2 * y**2 + 3 * x * y


# ========== 2. 设置初始值并启用梯度跟踪 ==========
x0 = torch.tensor(2.0, requires_grad=True)
y0 = torch.tensor(3.0, requires_grad=True)

print(f"初始点: x0 = {x0.item()}, y0 = {y0.item()}")

# ========== 3. 前向传播 ==========
output = f(x0, y0)

# ========== 4. 反向传播求梯度 ==========
output.backward()  # output 是标量，直接调用 backward()

# ========== 5. 获取梯度 ==========
grad_x_torch = x0.grad.item()
grad_y_torch = y0.grad.item()

print(f"\n=== PyTorch 自动求导结果 ===")
print(f"∂f/∂x = {grad_x_torch}")
print(f"∂f/∂y = {grad_y_torch}")
print(f"梯度 ∇f = ({grad_x_torch}, {grad_y_torch})")

# ========== 6. 手推结果对照 ==========
grad_x_manual = 3 * (2.0**2) + 3 * 3.0   # = 21
grad_y_manual = 4 * 3.0 + 3 * 2.0        # = 18

print(f"\n=== 手推梯度对照 ===")
print(f"∂f/∂x = {grad_x_manual}")
print(f"∂f/∂y = {grad_y_manual}")
print(f"梯度 ∇f = ({grad_x_manual}, {grad_y_manual})")

# ========== 7. 验证一致性 ==========
assert abs(grad_x_torch - grad_x_manual) < 1e-6
assert abs(grad_y_torch - grad_y_manual) < 1e-6
print(f"\n✅ 手推结果与 PyTorch 自动求导结果一致！")
```

### 3.4 运行输出

```
初始点: x0 = 2.0, y0 = 3.0

=== PyTorch 自动求导结果 ===
∂f/∂x = 21.0
∂f/∂y = 18.0
梯度 ∇f = (21.0, 18.0)

=== 手推梯度对照 ===
∂f/∂x = 21.0
∂f/∂y = 18.0
梯度 ∇f = (21.0, 18.0)

✅ 手推结果与 PyTorch 自动求导结果一致！
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
# 线性回归梯度下降示例
W = torch.randn(3, 1, requires_grad=True)  # 权重矩阵
lr = 0.01

for epoch in range(10):
    y_pred = X @ W                    # 前向
    loss = ((y_pred - y_true)**2).mean()  # MSE
    
    loss.backward()                   # 反向传播
    with torch.no_grad():
        W -= lr * W.grad              # 梯度下降更新
    W.grad.zero_()                    # 清零梯度
    
    print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")
```

---

## 五、常见陷阱

1. **忘记 `zero_grad()`**：梯度默认累加，不归零会导致梯度越来越大
2. **非标量 backward**：对向量/矩阵调用 `.backward()` 需要传入 `gradient` 参数
3. **`with torch.no_grad()`**：推理/评估时必须用，否则计算图不断累积，内存泄漏
4. **in-place 操作**：对 `requires_grad=True` 的叶子张量做 in-place 修改会破坏计算图
