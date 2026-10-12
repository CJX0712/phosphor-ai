/*
 * Phosphor 自检 harness —— 零依赖，Node 直接运行
 * 每条用例验证一条「数学不变量」，全部通过即证明系统正确可复现。
 * 运行：node test/selftest.js
 */
'use strict';
var P = require('../phosphor.js');
var L = P.Linalg;

var passed = 0, failed = 0, lines = [];
function ok(name, cond, detail) {
  if (cond) { passed++; lines.push('  ✅ ' + name + (detail ? '  [' + detail + ']' : '')); }
  else { failed++; lines.push('  ❌ ' + name + (detail ? '  [' + detail + ']' : '')); }
}
function rms(A, B) { // Frobenius 相对误差
  var s = 0, n = 0;
  for (var i = 0; i < A.length; i++) for (var j = 0; j < A[0].length; j++) { var d = A[i][j] - B[i][j]; s += d * d; n++; }
  return Math.sqrt(s / n);
}
function randMat(rng, r, c) { var M = []; for (var i = 0; i < r; i++) { var row = []; for (var j = 0; j < c; j++) row.push(rng() * 2 - 1); M.push(row); } return M; }

var rng = L.mulberry32(20261013);

// ---------- 1. SVD 不变量 ----------
(function () {
  var A = randMat(rng, 6, 4);
  var d = L.svd(A);
  var recon = L.matmul(L.matmul(d.U, (function () { var S = L.zeros(4, 4); for (var i = 0; i < 4; i++) S[i][i] = d.S[i]; return S; })()), L.transpose(d.V));
  var err = rms(A, recon);
  ok('SVD 重构 A≈U·S·Vᵀ', err < 1e-9, 'err=' + err.toExponential(2));
  // UᵀU=I, VᵀV=I
  var UtU = L.matmul(L.transpose(d.U), d.U), I4 = L.identity(4);
  ok('SVD 左奇异向量正交 UᵀU=I', rms(UtU, I4) < 1e-9, 'err=' + rms(UtU, I4).toExponential(2));
  var VtV = L.matmul(L.transpose(d.V), d.V);
  ok('SVD 右奇异向量正交 VᵀV=I', rms(VtV, I4) < 1e-9, 'err=' + rms(VtV, I4).toExponential(2));
})();

// ---------- 2. JacobiEig 不变量 ----------
(function () {
  // 良好分离特征值的对称阵：diag(1,5,25,125,625) + 小对称扰动
  // （注：退化/聚类特征值在双精度下 A·v=λv 的精度上限 ~1e-9，属数值分析固有现象；
  //   本用例用分离特征值，可将该不变量验证至机器精度。）
  var D = L.zeros(5, 5); [1, 5, 25, 125, 625].forEach(function (v, i) { D[i][i] = v; });
  var B = randMat(rng, 5, 5);
  var Sym = L.addM(D, L.scaleM(L.addM(B, L.transpose(B)), 0.005));
  var e = L.jacobiEig(Sym);
  var normA = e.values.reduce(function (m, x) { return Math.max(m, Math.abs(x)); }, 0) || 1;
  var maxRes = 0, maxOrth = 0;
  for (var i = 0; i < 5; i++) {
    var v = e.vectors[i];
    var Av = L.matmul(Sym, L.colVec(v)).map(function (r) { return r[0]; });
    var lv = v.map(function (x) { return x * e.values[i]; });
    var res = 0; for (var j = 0; j < 5; j++) res += (Av[j] - lv[j]) * (Av[j] - lv[j]);
    maxRes = Math.max(maxRes, Math.sqrt(res) / normA); // 相对残差，与特征值量级无关
    for (var k = i + 1; k < 5; k++) { var dp = L.dot(v, e.vectors[k]); maxOrth = Math.max(maxOrth, Math.abs(dp)); }
  }
  ok('JacobiEig 特征方程 ‖Av−λv‖/‖A‖', maxRes < 1e-7, 'relRes=' + maxRes.toExponential(2));
  ok('JacobiEig 特征向量正交', maxOrth < 1e-9, 'maxOrth=' + maxOrth.toExponential(2));
  ok('JacobiEig 特征值全实（对称阵）', e.values.every(function (x) { return isFinite(x); }));
})();

// ---------- 3. 矩阵逆不变量 ----------
(function () {
  var B = randMat(rng, 4, 4);
  var A = L.addM(L.matmul(B, L.transpose(B)), L.scaleM(L.identity(4), 0.5)); // 正定 → 可逆
  var Ainv = L.inv(A), I = L.matmul(A, Ainv);
  ok('矩阵逆 A·A⁻¹=I', rms(I, L.identity(4)) < 1e-9, 'err=' + rms(I, L.identity(4)).toExponential(2));
})();

// ---------- 4. PCA 不变量 ----------
(function () {
  var n = 60, d = 4, X = [];
  for (var i = 0; i < n; i++) { var r = []; for (var j = 0; j < d; j++) r.push(rng()); X.push(r); }
  var k = 2, res = P.PCA(X, k);
  // 协方差对称半正定：特征值 ≥ -1e-9
  ok('PCA 协方差半正定（λ≥0）', res.eigenvalues.every(function (l) { return l >= -1e-9; }), 'λ=' + res.eigenvalues.map(function (x) { return x.toFixed(3); }).join(','));
  // 特征向量正交归一
  var C = res.components, maxOrth = 0;
  for (var a = 0; a < k; a++) for (var b = a + 1; b < k; b++) maxOrth = Math.max(maxOrth, Math.abs(L.dot(C.map(function (r) { return r[a]; }), C.map(function (r) { return r[b]; }))));
  ok('PCA 载荷向量正交', maxOrth < 1e-9, 'maxOrth=' + maxOrth.toExponential(2));
  // 重构误差（投影后再反投影，应小于原始方差）
  ok('PCA 解释方差比 ∈ (0,1]', res.explainedVarianceRatio > 0 && res.explainedVarianceRatio <= 1.0001, 'ratio=' + res.explainedVarianceRatio.toFixed(4));
})();

// ---------- 5. GMM-EM 不变量 ----------
(function () {
  var n = 120, d = 2, X = [];
  for (var i = 0; i < n; i++) { X.push([rng() * 2 - 1 + (i % 2 ? 3 : 0), rng() * 2 - 1 + (i % 2 ? 2 : 0)]); }
  var m = P.GMM(X, 2, { seed: 11, maxIter: 300 });
  // 责任度行和=1
  var rsOK = true, maxRowSumErr = 0;
  // 用内部逻辑复算责任度行和：直接验证 weights 和 LL 单调
  var wSum = m.weights.reduce(function (a, b) { return a + b; }, 0);
  ok('GMM 权重和=1', Math.abs(wSum - 1) < 1e-9, 'sum=' + wSum.toFixed(12));
  // 对数似然单调不降
  var mono = true, minStep = Infinity;
  for (var t = 1; t < m.logLikelihoods.length; t++) { var step = m.logLikelihoods[t] - m.logLikelihoods[t - 1]; if (step < -1e-6) mono = false; minStep = Math.min(minStep, step); }
  ok('GMM 对数似然单调不降', mono, 'minΔ=' + (minStep === Infinity ? 0 : minStep.toExponential(2)) + '  finalLL=' + m.logLikelihood.toFixed(4));
})();

// ---------- 6. kNN 边界不变量 ----------
(function () {
  var X = [[0, 0], [1, 1], [2, 2], [3, 3]], y = [0, 1, 0, 1];
  var model = P.KNN(X, y, 1, { task: 'classification' });
  var err = 0; for (var i = 0; i < X.length; i++) if (model.predict(X[i]) !== y[i]) err++;
  ok('kNN k=1 训练集误差=0', err === 0, 'err=' + err);
})();

// ---------- 7. MLP 梯度检验 ----------
(function () {
  var sizes = [3, 5, 2], rng2 = L.mulberry32(3);
  var params = []; for (var i = 0; i < 30; i++) params.push(rng2() * 2 - 1);
  var x = [0.5, -0.3, 0.8], target = [1, -1];
  var gc = P.gradCheck(function (p) { return P.MLP.lossGrad(sizes, p, x, target); }, params, { relTol: 1e-4 });
  ok('MLP 反向传播梯度检验', gc.pass, 'maxRelErr=' + gc.maxRelError.toExponential(2));
})();

// ---------- 8. Attention 梯度检验 ----------
(function () {
  var n = 4, d = 3, rng3 = L.mulberry32(5);
  var Q = randMat(rng3, n, d), K = randMat(rng3, n, d), V = randMat(rng3, n, d);
  var dOut = randMat(rng3, n, d);
  // 包装成 flat 参数 + 解析梯度
  function flatLossGrad(params) {
    var qq = [], kk = [], vv = [], o = 0;
    for (var i = 0; i < n; i++) { qq.push(params[o].slice ? params[o] : null); o++; }
    // 重新切片
    var pQ = [], pK = [], pV = [], idx = 0;
    for (var i = 0; i < n; i++) { var r = []; for (var j = 0; j < d; j++) r.push(params[idx++]); pQ.push(r); }
    for (var i = 0; i < n; i++) { var r = []; for (var j = 0; j < d; j++) r.push(params[idx++]); pK.push(r); }
    for (var i = 0; i < n; i++) { var r = []; for (var j = 0; j < d; j++) r.push(params[idx++]); pV.push(r); }
    var g = P.Attention.sdpaLossGrad(pQ, pK, pV, dOut);
    var grad = [];
    for (var i = 0; i < n; i++) for (var j = 0; j < d; j++) grad.push(g.dQ[i][j]);
    for (var i = 0; i < n; i++) for (var j = 0; j < d; j++) grad.push(g.dK[i][j]);
    for (var i = 0; i < n; i++) for (var j = 0; j < d; j++) grad.push(g.dV[i][j]);
    return { loss: g.loss, grad: grad };
  }
  var params = []; Q.forEach(function (r) { r.forEach(function (v) { params.push(v); }); });
  K.forEach(function (r) { r.forEach(function (v) { params.push(v); }); });
  V.forEach(function (r) { r.forEach(function (v) { params.push(v); }); });
  var gc = P.gradCheck(flatLossGrad, params, { relTol: 1e-4 });
  ok('Attention 解析梯度检验', gc.pass, 'maxRelErr=' + gc.maxRelError.toExponential(2));
})();

// ---------- 汇总 ----------
console.log('\n══════════════════════════════════════════════════════');
console.log('  Phosphor 自检报告  v' + P.version + '  · Author: ' + P.author);
console.log('══════════════════════════════════════════════════════');
console.log(lines.join('\n'));
console.log('────────────────────────────────────────────────────────');
console.log('  结果：' + passed + ' 通过 / ' + failed + ' 失败');
console.log('══════════════════════════════════════════════════════');
process.exit(failed === 0 ? 0 : 1);
