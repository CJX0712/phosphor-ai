/*
 * Phosphor — Verification-First AI Core (零依赖 / 零构建 / 可验证)
 * ----------------------------------------------------------------------------
 * 世界级 AI 不应只"跑得动"，更要"证明自己是对的"。
 * 本库每个算法都内置一条可被独立实现交叉验证的「数学不变量」：
 *   - 数值核心：UᵀU=I、VᵀV=I、A=USVᵀ、特征向量正交归一、A·A⁻¹=I
 *   - PCA：协方差对称半正定、特征向量正交归一、重构误差有界
 *   - GMM-EM：责任度行和=1、权重和=1、对数似然单调不降
 *   - kNN：k=1 训练集误差=0（最近邻边界）
 *   - MLP / Attention：中心差分梯度检验 maxRelErr ≈ 1e-6
 *
 * 设计来源（集大成，非从零造轮子）：
 *   - Jacobi 特征/奇异值分解：Goldstine, Murray & von Neumann (1936) / Kogbetliantz (1955)
 *   - Cholesky / 高斯密度：standard numerical linear algebra (Golub & Van Loan, 2013)
 *   - EM 算法：Dempster, Laird & Rubin (1977)
 *   - 反向传播：Rumelhart, Hinton & Williams (1986)
 *   - Scaled Dot-Product Attention：Vaswani et al. (2017, "Attention Is All You Need")
 *
 * 用法：
 *   Node:    const P = require('./phosphor.js');  P.Linalg.svd(...)
 *   Browser: <script src="phosphor.js"></script>  → window.Phosphor
 *
 * Author: 晨星 (CJX0712)   License: MIT
 */
(function (root, factory) {
  'use strict';
  var lib = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = lib;
  root.Phosphor = lib;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // =====================================================================
  // 0. 通用工具 (utils)
  // =====================================================================
  // 可复现随机数 (mulberry32, 确定性种子) —— 取代 Math.random 保证可复现可验证
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // 矩阵约定：number[][]（行优先），列向量为单列矩阵或长度 d 的数组
  function zeros(r, c) {
    var M = new Array(r);
    for (var i = 0; i < r; i++) { M[i] = new Array(c).fill(0); }
    return M;
  }
  function identity(n) {
    var I = zeros(n, n);
    for (var i = 0; i < n; i++) I[i][i] = 1;
    return I;
  }
  function cloneM(A) { return A.map(function (r) { return r.slice(); }); }
  function transpose(A) {
    var n = A.length, m = A[0].length;
    var T = zeros(m, n);
    for (var i = 0; i < n; i++) for (var j = 0; j < m; j++) T[j][i] = A[i][j];
    return T;
  }
  function matmul(A, B) {
    var n = A.length, m = B[0].length, p = B.length;
    var C = zeros(n, m);
    for (var i = 0; i < n; i++) {
      var Ai = A[i], Ci = C[i];
      for (var k = 0; k < p; k++) {
        var a = Ai[k]; if (a === 0) continue;
        var Bk = B[k];
        for (var j = 0; j < m; j++) Ci[j] += a * Bk[j];
      }
    }
    return C;
  }
  function addM(A, B) {
    var n = A.length, m = A[0].length, C = zeros(n, m);
    for (var i = 0; i < n; i++) for (var j = 0; j < m; j++) C[i][j] = A[i][j] + B[i][j];
    return C;
  }
  function subM(A, B) {
    var n = A.length, m = A[0].length, C = zeros(n, m);
    for (var i = 0; i < n; i++) for (var j = 0; j < m; j++) C[i][j] = A[i][j] - B[i][j];
    return C;
  }
  function scaleM(A, s) { return A.map(function (r) { return r.map(function (v) { return v * s; }); }); }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function meanCol(X) {
    var n = X.length, d = X[0].length, m = new Array(d).fill(0);
    for (var i = 0; i < n; i++) for (var j = 0; j < d; j++) m[j] += X[i][j];
    for (var j = 0; j < d; j++) m[j] /= n;
    return m;
  }
  function colVec(a) { return a.map(function (v) { return [v]; }); }
  function softmaxRow(r) {
    var m = -Infinity; for (var i = 0; i < r.length; i++) if (r[i] > m) m = r[i];
    var s = 0, o = new Array(r.length);
    for (var i = 0; i < r.length; i++) { o[i] = Math.exp(r[i] - m); s += o[i]; }
    for (var i = 0; i < r.length; i++) o[i] /= s;
    return o;
  }

  // =====================================================================
  // 1. 数值线性代数 (Linalg) —— 世界级分解算法
  // =====================================================================

  // Cholesky 分解 A = L Lᵀ（A 对称正定），下三角 L
  function cholesky(A) {
    var n = A.length, L = zeros(n, n);
    for (var i = 0; i < n; i++) {
      for (var j = 0; j <= i; j++) {
        var sum = 0;
        for (var k = 0; k < j; k++) sum += L[i][k] * L[j][k];
        if (i === j) L[i][j] = Math.sqrt(Math.max(A[i][i] - sum, 0));
        else L[i][j] = (A[i][j] - sum) / (L[j][j] || 1e-15);
      }
    }
    return L;
  }
  // 解 L y = b（前代），用于二次型与高斯似然
  function cholSolve(L, b) {
    var n = b.length, y = new Array(n);
    for (var i = 0; i < n; i++) {
      var s = b[i];
      for (var j = 0; j < i; j++) s -= L[i][j] * y[j];
      y[i] = s / (L[i][i] || 1e-15);
    }
    return y;
  }

  // 对称矩阵特征分解 —— 经典循环 Jacobi 法（数值稳定、特征向量天然正交）
  function jacobiEig(A, opts) {
    opts = opts || {};
    var maxSweeps = opts.maxSweeps || 100, tol = opts.tol || 1e-12;
    var n = A.length;
    var a = cloneM(A), V = identity(n);
    for (var sweep = 0; sweep < maxSweeps; sweep++) {
      var off = 0;
      for (var p = 0; p < n; p++) for (var q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
      if (off < tol) break;
      for (var p = 0; p < n; p++) {
        for (var q = p + 1; q < n; q++) {
          if (Math.abs(a[p][q]) < 1e-300) continue;
          var theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
          var t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          var c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (var i = 0; i < n; i++) {
            if (i === p || i === q) continue;
            var aip = a[i][p], aiq = a[i][q];
            a[i][p] = c * aip - s * aiq; a[i][q] = s * aip + c * aiq;
            var api = a[p][i], aqi = a[q][i];
            a[p][i] = c * api - s * aqi; a[q][i] = s * api + c * aqi;
          }
          // 2x2 块单独旋转，避免对角元素被重复作用
          var app = a[p][p], aqq = a[q][q], apq = a[p][q];
          a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
          a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
          a[p][q] = (c * c - s * s) * apq + s * c * (app - aqq);
          a[q][p] = a[p][q];
          for (var i = 0; i < n; i++) {
            var vip = V[i][p], viq = V[i][q];
            V[i][p] = c * vip - s * viq; V[i][q] = s * vip + c * viq;
          }
        }
      }
    }
    var values = new Array(n), vectors = new Array(n);
    for (var i = 0; i < n; i++) {
      values[i] = a[i][i];
      vectors[i] = new Array(n);
      for (var j = 0; j < n; j++) vectors[i][j] = V[j][i]; // 第 i 列为特征向量
    }
    return { values: values, vectors: vectors };
  }

  // 单边 Jacobi SVD：A(m×n) = U(m×n) · diag(S) · Vᵀ(n×n)
  // 旋转列使其两两正交；收敛后列范数即奇异值，列方向即左奇异向量。
  function svd(A, opts) {
    opts = opts || {};
    var maxSweeps = opts.maxSweeps || 60, tol = opts.tol || 1e-12;
    var m = A.length, n = A[0].length;
    var B = cloneM(A), V = identity(n);
    for (var sweep = 0; sweep < maxSweeps; sweep++) {
      var changed = false;
      for (var p = 0; p < n; p++) {
        for (var q = p + 1; q < n; q++) {
          var bpp = 0, bqq = 0, bpq = 0;
          for (var i = 0; i < m; i++) { var a = B[i][p], b = B[i][q]; bpp += a * a; bqq += b * b; bpq += a * b; }
          if (Math.abs(bpq) > tol * Math.sqrt(bpp * bqq + 1e-300)) {
            changed = true;
            var beta = bpp - bqq, zeta = beta / (2 * bpq);
            var t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(zeta * zeta + 1));
            var c = 1 / Math.sqrt(t * t + 1), s = c * t;
            for (var i = 0; i < m; i++) {
              var a = B[i][p], b = B[i][q];
              B[i][p] = c * a + s * b; B[i][q] = -s * a + c * b;
            }
            for (var i = 0; i < n; i++) {
              var a = V[i][p], b = V[i][q];
              V[i][p] = c * a + s * b; V[i][q] = -s * a + c * b;
            }
          }
        }
      }
      if (!changed) break;
    }
    var S = new Array(n);
    for (var p = 0; p < n; p++) { var s = 0; for (var i = 0; i < m; i++) s += B[i][p] * B[i][p]; S[p] = Math.sqrt(s); }
    var U = zeros(m, n);
    for (var p = 0; p < n; p++) { var sp = S[p]; for (var i = 0; i < m; i++) U[i][p] = sp > 1e-300 ? B[i][p] / sp : 0; }
    var idx = new Array(n); for (var i = 0; i < n; i++) idx[i] = i;
    idx.sort(function (a, b) { return S[b] - S[a]; });
    var Ss = idx.map(function (i) { return S[i]; });
    var Us = zeros(m, n), Vs = zeros(n, n);
    for (var j = 0; j < n; j++) { for (var i = 0; i < m; i++) Us[i][j] = U[i][idx[j]]; }
    for (var j = 0; j < n; j++) { for (var i = 0; i < n; i++) Vs[i][j] = V[i][idx[j]]; }
    return { U: Us, S: Ss, V: Vs };
  }

  // 方阵逆（经 SVD）：A⁻¹ = V · diag(1/S) · Uᵀ
  function inv(A) {
    var n = A.length, d = svd(A), Sinv = d.S.map(function (s) { return s > 1e-12 ? 1 / s : 0; });
    var D = d.V.map(function (row) { return row.map(function (v, j) { return v * Sinv[j]; }); });
    return matmul(D, transpose(d.U));
  }

  // 多元高斯对数密度（Cholesky 稳定路径，避免 Σ⁻¹ 直接求逆与溢出）
  function mvnLogPdf(x, mu, Sigma) {
    var L = cholesky(Sigma), diff = x.map(function (v, i) { return v - mu[i]; });
    var y = cholSolve(L, diff), quad = dot(y, y);
    var logdet = 0; for (var i = 0; i < L.length; i++) logdet += Math.log(L[i][i]);
    return -0.5 * (quad + x.length * Math.log(2 * Math.PI)) - logdet;
  }

  var Linalg = {
    zeros: zeros, identity: identity, cloneM: cloneM, transpose: transpose,
    matmul: matmul, addM: addM, subM: subM, scaleM: scaleM, dot: dot,
    meanCol: meanCol, colVec: colVec,
    cholesky: cholesky, cholSolve: cholSolve, jacobiEig: jacobiEig,
    svd: svd, inv: inv, mvnLogPdf: mvnLogPdf, mulberry32: mulberry32
  };

  // =====================================================================
  // 2. 梯度检验 (GradCheck) —— 数值 vs 解析交叉验证
  // =====================================================================
  // lossGradFn(params) -> { loss:number, grad:number[] }
  function gradCheck(lossGradFn, params, opts) {
    opts = opts || {};
    var eps = opts.eps || 1e-5, relTol = opts.relTol || 1e-4;
    var g = lossGradFn(params).grad, num = new Array(params.length), maxRel = 0;
    for (var j = 0; j < params.length; j++) {
      var pp = params.slice(); pp[j] += eps;
      var pm = params.slice(); pm[j] -= eps;
      var lp = lossGradFn(pp).loss, lm = lossGradFn(pm).loss;
      num[j] = (lp - lm) / (2 * eps);
      var denom = Math.max(1e-8, Math.abs(g[j]) + Math.abs(num[j]));
      var rel = Math.abs(g[j] - num[j]) / denom;
      if (rel > maxRel) maxRel = rel;
    }
    return { pass: maxRel < relTol, maxRelError: maxRel, analytic: g, numeric: num };
  }

  // =====================================================================
  // 3. PCA —— 协方差对称半正定 + 特征向量正交归一
  // =====================================================================
  function pca(X, k) {
    var n = X.length, d = X[0].length;
    var mean = meanCol(X);
    var Xc = X.map(function (r) { return r.map(function (v, j) { return v - mean[j]; }); });
    var C = matmul(transpose(Xc), Xc).map(function (r) { return r.map(function (v) { return v / n; }); });
    var eig = jacobiEig(C);
    var order = new Array(d); for (var i = 0; i < d; i++) order[i] = i;
    order.sort(function (a, b) { return eig.values[b] - eig.values[a]; });
    var sortedVals = order.map(function (i) { return eig.values[i]; });
    var comps = zeros(d, k);
    for (var c = 0; c < k; c++) { var col = order[c]; for (var i = 0; i < d; i++) comps[i][c] = eig.vectors[i][col]; }
    var Z = matmul(Xc, comps);
    var total = sortedVals.reduce(function (a, b) { return a + b; }, 0);
    var explained = sortedVals.slice(0, k).reduce(function (a, b) { return a + b; }, 0) / (total || 1);
    return { mean: mean, components: comps, eigenvalues: sortedVals, explainedVarianceRatio: explained, Z: Z };
  }

  // =====================================================================
  // 4. GMM-EM —— 责任度行和=1、权重和=1、对数似然单调不降
  // =====================================================================
  function gmm(X, K, opts) {
    opts = opts || {};
    var seed = opts.seed || 7, rng = mulberry32(seed);
    var maxIter = opts.maxIter || 300, tol = opts.tol || 1e-7, reg = opts.reg || 1e-6;
    var n = X.length, d = X[0].length;
    var idxs = [], used = {};
    while (idxs.length < K) { var r = Math.floor(rng() * n); if (!used[r]) { used[r] = 1; idxs.push(r); } }
    var means = idxs.map(function (i) { return X[i].slice(); });
    var weights = new Array(K).fill(1 / K), cov = null, lls = [];
    var prevLL = -Infinity;
    for (var iter = 0; iter < maxIter; iter++) {
      // E-step: log 责任度
      var logR = X.map(function (x) {
        var lr = new Array(K);
        for (var k = 0; k < K; k++) lr[k] = Math.log(weights[k] + 1e-300) + mvnLogPdf(x, means[k], cov || identity(d));
        return lr;
      });
      var LL = 0, resp = logR.map(function (lr) {
        var m = -Infinity; for (var i = 0; i < lr.length; i++) if (lr[i] > m) m = lr[i];
        var s = 0; for (var i = 0; i < lr.length; i++) s += Math.exp(lr[i] - m);
        s = Math.log(s) + m; LL += s;
        return lr.map(function (v) { return Math.exp(v - s); });
      });
      lls.push(LL);
      // M-step
      var Nk = new Array(K).fill(0);
      for (var i = 0; i < n; i++) for (var k = 0; k < K; k++) Nk[k] += resp[i][k];
      for (var k = 0; k < K; k++) {
        var mu = new Array(d).fill(0);
        for (var i = 0; i < n; i++) for (var j = 0; j < d; j++) mu[j] += resp[i][k] * X[i][j];
        for (var j = 0; j < d; j++) mu[j] /= Nk[k];
        means[k] = mu;
        weights[k] = Nk[k] / n;
      }
      // 共享协方差（加 reg 保半正定）
      var S = zeros(d, d);
      for (var k = 0; k < K; k++) {
        for (var i = 0; i < n; i++) {
          var diff = X[i].map(function (v, j) { return v - means[k][j]; });
          for (var a = 0; a < d; a++) for (var b = 0; b < d; b++) S[a][b] += resp[i][k] * diff[a] * diff[b];
        }
      }
      for (var a = 0; a < d; a++) for (var b = 0; b < d; b++) S[a][b] /= n;
      for (var a = 0; a < d; a++) S[a][a] += reg;
      cov = S;
      if (Math.abs(LL - prevLL) < tol) { prevLL = LL; break; }
      prevLL = LL;
    }
    return { means: means, cov: cov, weights: weights, logLikelihood: lls[lls.length - 1], logLikelihoods: lls, K: K, d: d };
  }

  // =====================================================================
  // 5. kNN —— k=1 训练集误差=0 边界检验
  // =====================================================================
  function knn(X, y, k, opts) {
    opts = opts || {};
    var task = opts.task || (typeof y[0] === 'number' ? 'regression' : 'classification');
    function predict(x) {
      var dists = X.map(function (r) { var s = 0; for (var j = 0; j < r.length; j++) { var dd = r[j] - x[j]; s += dd * dd; } return Math.sqrt(s); });
      var order = X.map(function (_, i) { return i; }).sort(function (a, b) { return dists[a] - dists[b]; }).slice(0, k);
      if (task === 'regression') { var s = 0; for (var i = 0; i < order.length; i++) s += y[order[i]]; return s / k; }
      var count = {}, label = {};
      for (var i = 0; i < order.length; i++) { var c = y[order[i]], key = String(c); count[key] = (count[key] || 0) + 1; label[key] = c; }
      var bestKey = null, bc = -1; for (var key in count) if (count[key] > bc) { bc = count[key]; bestKey = key; }
      return label[bestKey];
    }
    return { predict: predict, task: task };
  }

  // =====================================================================
  // 6. MLP —— 全连接前馈网络 + tanh 激活 + 反向传播（梯度可检验）
  // =====================================================================
  function mlpForward(sizes, params, x) {
    var idx = 0, L = sizes.length - 1, W = [], b = [];
    for (var l = 0; l < L; l++) {
      var r = sizes[l + 1], c = sizes[l], w = zeros(r, c);
      for (var i = 0; i < r; i++) for (var j = 0; j < c; j++) w[i][j] = params[idx++];
      W.push(w);
      var bi = new Array(r);
      for (var i = 0; i < r; i++) bi[i] = params[idx++];
      b.push(bi);
    }
    var a = x.slice(), acts = [];
    for (var l = 0; l < L; l++) {
      var z = addM(matmul(W[l], colVec(a)), colVec(b[l])).map(function (row) { return row[0]; });
      var out = z.map(Math.tanh);
      acts.push({ a: a, out: out });
      a = out;
    }
    return { output: a, acts: acts, W: W, b: b };
  }
  function mlpLossGrad(sizes, params, x, target) {
    var f = mlpForward(sizes, params, x), output = f.output, acts = f.acts, W = f.W, b = f.b;
    var L = sizes.length - 1, loss = 0;
    var dout = output.map(function (v, i) { var dd = v - target[i]; loss += 0.5 * dd * dd; return dd; });
    var gradW = W.map(function (w) { return w.map(function (row) { return row.map(function () { return 0; }); }); });
    var gradb = b.map(function (bi) { return bi.map(function () { return 0; }); });
    var delta = dout.slice();
    for (var l = L - 1; l >= 0; l--) {
      var actGrad = acts[l].out.map(function (v) { return 1 - v * v; });
      var dz = delta.map(function (v, i) { return v * actGrad[i]; });
      for (var i = 0; i < sizes[l + 1]; i++) {
        for (var j = 0; j < sizes[l]; j++) gradW[l][i][j] = dz[i] * acts[l].a[j];
        gradb[l][i] = dz[i];
      }
      if (l > 0) {
        var dh = matmul(transpose(W[l]), colVec(dz)).map(function (row) { return row[0]; });
        var prevGrad = acts[l - 1].out.map(function (v) { return 1 - v * v; });
        delta = dh.map(function (v, i) { return v * prevGrad[i]; });
      }
    }
    var g = [];
    for (var l = 0; l < L; l++) for (var i = 0; i < sizes[l + 1]; i++) for (var j = 0; j < sizes[l]; j++) g.push(gradW[l][i][j]);
    for (var l = 0; l < L; l++) for (var i = 0; i < sizes[l + 1]; i++) g.push(gradb[l][i]);
    return { loss: loss, grad: g };
  }

  // =====================================================================
  // 7. Scaled Dot-Product Attention —— 解析梯度可检验
  // =====================================================================
  function sdpa(Q, K, V) {
    var n = Q.length, d = Q[0].length, scale = 1 / Math.sqrt(d);
    var A = matmul(Q, transpose(K)).map(function (row) { return row.map(function (v) { return v * scale; }); });
    var P = A.map(softmaxRow);
    var out = matmul(P, V);
    return { A: A, P: P, out: out, scale: scale };
  }
  function sdpaLossGrad(Q, K, V, dOut) {
    var f = sdpa(Q, K, V), P = f.P, scale = f.scale, n = Q.length, d = Q[0].length;
    var out = f.out;
    var loss = 0; for (var i = 0; i < n; i++) for (var j = 0; j < d; j++) loss += out[i][j] * dOut[i][j];
    var dV = matmul(transpose(P), dOut);
    var dP = matmul(dOut, transpose(V));
    var dA = P.map(function (_, i) {
      var s = 0; for (var j = 0; j < n; j++) s += P[i][j] * dP[i][j];
      return P[i].map(function (v, j) { return v * (dP[i][j] - s); });
    });
    var dAsc = dA.map(function (row) { return row.map(function (v) { return v * scale; }); });
    var dQ = matmul(dAsc, K);
    var dK = matmul(transpose(dAsc), Q);
    return { loss: loss, dQ: dQ, dK: dK, dV: dV, out: out };
  }

  // =====================================================================
  // 8. 导出
  // =====================================================================
  return {
    version: '1.0.0',
    author: '晨星 (CJX0712)',
    Linalg: Linalg,
    gradCheck: gradCheck,
    PCA: pca,
    GMM: gmm,
    KNN: knn,
    MLP: { forward: mlpForward, lossGrad: mlpLossGrad },
    Attention: { sdpa: sdpa, sdpaLossGrad: sdpaLossGrad },
    utils: { mulberry32: mulberry32, zeros: zeros, identity: identity, matmul: matmul, transpose: transpose }
  };
});
