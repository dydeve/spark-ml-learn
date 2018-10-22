### Classification and Regression - 分类与回归
---
`spark.mllib`包提供多种方法用于[binary classification(二分类)](http://en.wikipedia.org/wiki/Binary_classification),[multiclass classification(多分类)](http://en.wikipedia.org/wiki/Multiclass_classification),[regression analysis(回归分析)](http://en.wikipedia.org/wiki/Regression_analysis).下表展示了每种问题的支持算法.

问题类型 | 支持方法
---|---
二分类 | 线性SVMs，逻辑回归，决策树，随机森林，梯度增强树，朴素贝叶斯
多分类 | 逻辑回归，决策树，随机森林，朴素贝叶斯
回归 | linear least squares(线性最小二乘法), Lasso, ridge regression, 决策树, 随机森林, 梯度增强树, isotonic regression(保序回归)

更多细节，请点击以下链接
- [线性模型](mllib-linear-methods.md)
    - [分类(支持向量机,逻辑回归)]()
    - [线性回归(最小二乘法,Lasso,ridge)]()
- [决策树](mllib-decision-tree.md)
- [组合数](mllib-ensembles.md)
    - [随机森林]()
    - [梯度增强树]()
- [朴素贝叶斯](mllib-naive-bayes.md)
- [Isotonic regression](mllib-isotonic-regression.md)









