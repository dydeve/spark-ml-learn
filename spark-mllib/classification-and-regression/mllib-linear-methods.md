# Linear Methods(线性方法)
---
- <a href="#mathematical-formulation">Mathematical formulation(数学公式)</a>
    - <a href="#loss-functions">Loss functions(损失函数)</a>
    - <a href="#regularizers">Regularizers(正则化)</a>
    - <a href="#optimization">Optimization(优化)</a>
- <a href="#classification">Classification(分类)</a>
    - <a href="#svm">Linear Support Vector Machines (SVMs)支持向量机</a>
    - <a href="#logistic-regression">Logistic regression(逻辑回归)</a>
- <a href="#regression">Regression(回归)</a>
    - <a href="#linear-least-squares_lasso_ridge-regression">Linear least squares, Lasso, and ridge regression(线性最小二乘，套索和岭回归)</a>
    - <a href="#streaming-linear-regression">Streaming linear regression(流式线性回归)</a>
- <a href="#implementation-developer">Implementation (developer)(实施 开发者)</a>

### <a id="mathematical-formulation">Mathematical formulation(数学公式)</a>
许多标准的机器学习算法可以公式化为凸优化问题(`convex optimization problem`),例如找到凸函数<code>$f$</code>，该函数依赖于可变向量<code>$\wv$</code> (called <code>weights</code> in the code)




<code>\begin{equation}
    f(\wv) := \lambda\, R(\wv) +
    \frac1n \sum_{i=1}^n L(\wv;\x_i,y_i)
    \label{eq:regPrimal}
    \ .
\end{equation}</code>





