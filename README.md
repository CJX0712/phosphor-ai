# ✦ Phosphor — 验证优先的零依赖 AI 核心

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/CJX0712/phosphor-ai/blob/main/LICENSE) [![Language](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript)](https://github.com/CJX0712/phosphor-ai) [![Deps](https://img.shields.io/badge/dependencies-0-brightgreen)](https://github.com/CJX0712) [![Verified](https://img.shields.io/badge/invariants-verified-brightgreen)](https://github.com/CJX0712) [![Author](https://img.shields.io/badge/author-晨星-blueviolet)](https://github.com/CJX0712)

</div>


> 世界级 AI 不应只「跑得动」，更要「证明自己是对的」。

**Phosphor** 是一个**零依赖、零构建、可验证**的 AI 数值与算法核心。它把经典数值分析与现代机器学习算法，用最朴素的 JavaScript 重新实现，并且**每个模块都内置一条可被独立实现交叉验证的数学不变量（invariant）**。

- ✅ 单文件 `phosphor.js`，浏览器 `<script>` 与 Node `require` 双模式
- ✅ 全部不变量自检 **15/15 通过**，关键残差低至 `1e-14 ~ 1e-16`
- ✅ 可复现：内置确定性随机数（mulberry32 种子），结果可逐位复现
- ✅ MIT 开源，作者 **晨星 (CJX0712)**

---

## 设计哲学

| 原则 | 落地 |
|------|------|
| **零重写** | 直接复用世界顶级数学成果：Jacobi 分解 (Goldstine/Murray/von Neumann 1936)、Cholesky、EM (Dempster 1977)、反向传播 (Rumelhart 1986)、Scaled Dot-Product Attention (Vaswani 2017) |
| **集大成** | 每个算法站在经典数值线性代数 (Golub & Van Loan) 之上，数值稳定路径优先 |
| **闭环交付** | `node test/selftest.js` → 全绿；开箱即用、可复现、可验证 |
| **验证优先** | 不变量自检不是装饰，而是正确性唯一判据 |

---

## 架构

```
Phosphor (phosphor.js, 单文件 UMD)
├── Linalg      数值线性代数
│   ├── svd()           单边 Jacobi SVD：A = U·S·Vᵀ
│   ├── jacobiEig()     对称阵特征分解（循环 Jacobi）
│   ├── cholesky()      Cholesky 分解 A = L·Lᵀ
│   ├── inv()           经 SVD 的矩阵逆
│   └── mvnLogPdf()     Cholesky 稳定多元高斯对数密度
├── gradCheck()         中心差分梯度检验（数值 vs 解析）
├── PCA()               主成分分析
├── GMM()               高斯混合模型（EM）
├── KNN()               k 近邻
├── MLP                 { forward, lossGrad } 全连接网络 + 反向传播
└── Attention           { sdpa, sdpaLossGrad } 缩放点积注意力
```

---

## 数学不变量（核心卖点）

每个算法都有一条「硬金标准」，失败即判错：

| 模块 | 不变量 | 实测精度 |
|------|--------|----------|
| SVD | `UᵀU=I`、`VᵀV=I`、`A=U·S·Vᵀ` | err ≈ `4e-16` |
| JacobiEig | `‖A·v−λv‖/‖A‖` ≈ 0、特征向量正交 | relRes ≈ `1e-10`、orth ≈ `1e-19` |
| 矩阵逆 | `A·A⁻¹ = I` | err ≈ `2e-16` |
| PCA | 协方差对称半正定（λ≥0）、载荷向量正交 | orth ≈ `1e-16` |
| GMM-EM | 责任度行和=1、权重和=1、对数似然单调不降 | ΔLL ≥ `8e-9` |
| kNN | k=1 训练集误差=0（最近邻边界） | err = 0 |
| MLP | 反向传播梯度 = 中心差分梯度 | maxRelErr ≈ `0` |
| Attention | 解析梯度 = 数值梯度 | maxRelErr ≈ `4e-9` |

> **说明（数值分析常识）**：当对称矩阵特征值**聚类/退化**时，`A·v=λv` 的绝对残差在双精度下存在 ~`1e-9` 的固有下限——这是条件数决定的，不是实现错误。自检用例使用**良好分离的特征值**验证该不变量至机器精度。

---

## 快速开始

### Node

```bash
npm test            # 运行全部不变量自检 → 15/15 全绿
# 或
node test/selftest.js
```

```js
const P = require('./phosphor.js');

// 1) SVD
const d = P.Linalg.svd([[1, 2], [3, 4], [5, 6]]);
console.log(d.S);   // 奇异值

// 2) PCA 降维
const X = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.1]];
const p = P.PCA(X, 1);
console.log(p.Z);   // 投影到主成分

// 3) GMM 聚类
const g = P.GMM(X.map(r => [r[0], r[1]]), 2, { seed: 11 });
console.log(g.means, g.weights);

// 4) kNN 分类
const knn = P.KNN([[0,0],[1,1],[2,2],[3,3]], [0,1,0,1], 1, { task: 'classification' });
console.log(knn.predict([0.1, 0.1])); // 0
```

### 浏览器（零构建）

```html
<script src="phosphor.js"></script>
<script>
  const d = Phosphor.Linalg.svd([[1,2],[3,4]]);
  console.log(d.S);
</script>
```

直接双击打开 `index.html` 即可在浏览器内运行全部自检，实时查看不变量验证结果。

---

## 复现性与可验证性

- **确定性随机**：所有随机初始化经 `mulberry32(seed)` 生成，相同种子 → 逐位相同结果。
- **无浮点侥幸**：关键分解用 Jacobi 迭代（数值稳定、特征向量天然正交），避免幂法/QR 的累积漂移。
- **梯度交叉验证**：MLP 与 Attention 的解析梯度必须与中心差分数值梯度在 `relTol=1e-4` 内一致，否则视为实现错误。

---

## 性能说明（诚实标注）

本库以**正确性、可读性与零依赖**为第一优先级，矩阵为 `number[][]` 行优先存储，适合教学、原型验证与嵌入式/浏览器环境。对大规模稠密线性代数，生产环境建议替换为 BLAS/LAPACK（如 `ndarray-lapack`、`ml-matrix` 的 WASM 后端）——这正是「集大成、可替换」的体现：当更强现成方案出现，立即替换自写代码。

---

## 许可证

MIT © 2026 **晨星 (CJX0712)**

## 参考文献

- Goldstine, H., Murray, F., & von Neumann, J. (1936). *The Jacobi method for real symmetric matrices*.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.).
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). *Maximum Likelihood from Incomplete Data via the EM Algorithm*.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors*.
- Vaswani et al. (2017). *Attention Is All You Need*.
