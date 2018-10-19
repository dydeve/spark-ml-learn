### Stratified sampling 分层取样
#### 简介
不像`spark.mllib`里的其他统计函数，分层取样方法`sampleByKey`、`sampleByKeyExact`可以在 key-value RDD 上运行。在分层取样中，key可以认为是`label`，value是具体的属性。举例，key可以是男人、女人，或文档IDs，值可以是年龄列表或文档中的词列表。`sampleByKey`将决定一个数据是否是采样数据，因此需要传递数据并提供期望的采样大小.比起在每层上使用`sampleByKey`进行随机抽样，`sampleByKeyExact`需要更资源，但它能提供精准抽样，可信度99.99%(目前不支持`Python`)。

`sampleByKeyExact`进行精准抽样`⌈fk⋅nk⌉∀k∈K`，`fk`是需要获取key为k的比例，`nk`是以k为键的key-value对的数量，`K`是键的集合。无需替换的采样需要`RDD`上的一次额外`pass over`，需要替换的采样需要两次额外的`pass`(Sampling without replacement requires one additional pass over the RDD to guarantee sample size, whereas sampling with replacement requires two additional passes.)
```scala
// an RDD[(K, V)] of any key value pairs
val data = sc.parallelize(
  Seq((1, 'a'), (1, 'b'), (2, 'c'), (2, 'd'), (2, 'e'), (3, 'f')))

// specify the exact fraction desired from each key
val fractions = Map(1 -> 0.1, 2 -> 0.6, 3 -> 0.3)

// Get an approximate sample from each stratum
val approxSample = data.sampleByKey(withReplacement = false, fractions = fractions)
// Get an exact sample from each stratum
val exactSample = data.sampleByKeyExact(withReplacement = false, fractions = fractions)
```
详见`examples/src/main/scala/org/apache/spark/examples/mllib/StratifiedSamplingExample.scala`

```text
approxSample size is 2
(2,d)
(2,e)

exactSample its size is 4
(1,b)
(2,d)
(2,e)
(3,f)
```

#### 解析
```scala
 /**
   * Return a subset of this RDD sampled by key (via stratified sampling).
   *
   * Create a sample of this RDD using variable sampling rates for different keys as specified by
   * `fractions`, a key to sampling rate map, via simple random sampling with one pass over the
   * RDD, to produce a sample of size that's approximately equal to the sum of
   * math.ceil(numItems * samplingRate) over all key values.
   *
   * @param withReplacement whether to sample with or without replacement
   * @param fractions map of specific keys to sampling rates
   * @param seed seed for the random number generator
   * @return RDD containing the sampled subset
   */
  def sampleByKey(withReplacement: Boolean,
      fractions: Map[K, Double],
      seed: Long = Utils.random.nextLong): RDD[(K, V)] = self.withScope {

    require(fractions.values.forall(v => v >= 0.0), "Negative sampling rates.")

    val samplingFunc = if (withReplacement) {
      //泊松取样
      StratifiedSamplingUtils.getPoissonSamplingFunction(self, fractions, false, seed)
    } else {
      //伯努利取样
      StratifiedSamplingUtils.getBernoulliSamplingFunction(self, fractions, false, seed)
    }
    //构造MapPartitionsRDD
    self.mapPartitionsWithIndex(samplingFunc, preservesPartitioning = true)
  }
  
 /**
   * Return a subset of this RDD sampled by key (via stratified sampling) containing exactly
   * math.ceil(numItems * samplingRate) for each stratum (group of pairs with the same key).
   *
   * This method differs from [[sampleByKey]] in that we make additional passes over the RDD to
   * create a sample size that's exactly equal to the sum of math.ceil(numItems * samplingRate)
   * over all key values with a 99.99% confidence. When sampling without replacement, we need one
   * additional pass over the RDD to guarantee sample size; when sampling with replacement, we need
   * two additional passes.
   *
   * @param withReplacement whether to sample with or without replacement
   * @param fractions map of specific keys to sampling rates
   * @param seed seed for the random number generator
   * @return RDD containing the sampled subset
   */
  def sampleByKeyExact(
      withReplacement: Boolean,
      fractions: Map[K, Double],
      seed: Long = Utils.random.nextLong): RDD[(K, V)] = self.withScope {

    require(fractions.values.forall(v => v >= 0.0), "Negative sampling rates.")

    val samplingFunc = if (withReplacement) {
      StratifiedSamplingUtils.getPoissonSamplingFunction(self, fractions, true, seed)
    } else {
      StratifiedSamplingUtils.getBernoulliSamplingFunction(self, fractions, true, seed)
    }
    self.mapPartitionsWithIndex(samplingFunc, preservesPartitioning = true)
  }
```
`sampleByKey`、`sampleByKeyExact`方法均有`withReplacement`方法，

withReplacement:
 - true: Poisson - 泊松分布 取样
 - false: Bernoulli - 伯努利分布 取样

`getPoissonSamplingFunction`，`getBernoulliSamplingFunction`均有`exact: Boolean`入参，决定时候精确抽样

withReplacement=true，泊松抽样
```scala
 /**
   * Return the per partition sampling function used for sampling with replacement.
   *
   * When exact sample size is required, we make two additional passed over the RDD to determine
   * the exact sampling rate that guarantees sample size with high confidence. The first pass
   * counts the number of items in each stratum (group of items with the same key) in the RDD, and
   * the second pass uses the counts to determine exact sampling rates.
   *
   * The sampling function has a unique seed per partition.
   */
  def getPoissonSamplingFunction[K: ClassTag, V: ClassTag](rdd: RDD[(K, V)],
      fractions: Map[K, Double],
      exact: Boolean,
      seed: Long): (Int, Iterator[(K, V)]) => Iterator[(K, V)] = {
    // TODO implement the streaming version of sampling w/ replacement that doesn't require counts
    if (exact) {//sampleByKeyExact
      val counts = Some(rdd.countByKey())
      //计算立即接受的样本数量，并且为每层生成候选名单
      val finalResult = getAcceptanceResults(rdd, true, fractions, counts, seed)
      ////决定接受样本的阈值，生成准确的样本大小
      val thresholdByKey = computeThresholdByKey(finalResult, fractions)
      (idx: Int, iter: Iterator[(K, V)]) => {
        val rng = new RandomDataGenerator()
        rng.reSeed(seed + idx)
        iter.flatMap { item =>
          val key = item._1
          val acceptBound = finalResult(key).acceptBound
          // Must use the same invoke pattern on the rng as in getSeqOp for with replacement
          // in order to generate the same sequence of random numbers when creating the sample
          val copiesAccepted = if (acceptBound == 0) 0L else rng.nextPoisson(acceptBound)
          //候选名单
          val copiesWaitlisted = rng.nextPoisson(finalResult(key).waitListBound)
          val copiesInSample = copiesAccepted +
            (0 until copiesWaitlisted).count(i => rng.nextUniform() < thresholdByKey(key))
          if (copiesInSample > 0) {
            Iterator.fill(copiesInSample.toInt)(item)
          } else {
            Iterator.empty
          }
        }
      }
    } else {//sampleByKey
      (idx: Int, iter: Iterator[(K, V)]) => {
        //随机数生成器，支持均匀值-uniform values、泊松值-Poisson values
        val rng = new RandomDataGenerator()
        rng.reSeed(seed + idx)
        iter.flatMap { item =>
          val count = rng.nextPoisson(fractions(item._1))
          if (count == 0) {
            Iterator.empty
          } else {
            Iterator.fill(count)(item)
          }
        }
      }
    }
  }
```

withReplacement=false，伯努利抽样
```scala
 /**
   * Return the per partition sampling function used for sampling without replacement.
   *
   * When exact sample size is required, we make an additional pass over the RDD to determine the
   * exact sampling rate that guarantees sample size with high confidence.
   *
   * The sampling function has a unique seed per partition.
   */
  def getBernoulliSamplingFunction[K, V](rdd: RDD[(K, V)],
      fractions: Map[K, Double],
      exact: Boolean,
      seed: Long): (Int, Iterator[(K, V)]) => Iterator[(K, V)] = {
    var samplingRateByKey = fractions
    if (exact) {//sampleByKeyExact
      // determine threshold for each stratum and resample
      //计算立即接受的样本数量，并且为每层生成候选名单
      val finalResult = getAcceptanceResults(rdd, false, fractions, None, seed)
      //决定接受样本的阈值，生成准确的样本大小
      samplingRateByKey = computeThresholdByKey(finalResult, fractions)
    }
    (idx: Int, iter: Iterator[(K, V)]) => {
      val rng = new RandomDataGenerator()
      rng.reSeed(seed + idx)
      // Must use the same invoke pattern on the rng as in getSeqOp for without replacement
      // in order to generate the same sequence of random numbers when creating the sample
      iter.filter(t => rng.nextUniform() < samplingRateByKey(t._1))
    }
  }
```

参数 | 含义
---|---
withReplacement | 每次抽样是否有放回
fractions | 控制不同key的抽样率
seed | 随机数种子

### 参考
[1][endymecy's github](https://github.com/endymecy/spark-ml-source-analysis/blob/master/%E5%9F%BA%E6%9C%AC%E7%BB%9F%E8%AE%A1/tratified-sampling.md)

[2][spark mllib stratified-sampling](http://spark.apache.org/docs/latest/mllib-statistics.html#stratified-sampling)

### 拓展阅读：

[1][泊松分佈](https://zh.wikipedia.org/wiki/%E6%B3%8A%E6%9D%BE%E5%88%86%E4%BD%88)

[2][伯努利分布](https://zh.wikipedia.org/wiki/%E4%BC%AF%E5%8A%AA%E5%88%A9%E5%88%86%E5%B8%83)



