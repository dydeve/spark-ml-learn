### Random data generation 随机数生成

随机数生成，在随机算法、原型(prototyping)、性能测试中很有用。`spark.mllib`支持生成随机RDD，RDD的` i.i.d. values`来自于给定的分布:均匀分布、标准正太分布、泊松分布(uniform, standard normal, or Poisson.)

`RandomRDDs`提供工厂方法生成随机double RDDs 或 vector RDDs。以下案例生成随机double RDD，它的值服从标准正太分布`N(0, 1)`,然后映射到`N(1, 4)`
```scala
import org.apache.spark.SparkContext
import org.apache.spark.mllib.random.RandomRDDs._

val sc: SparkContext = ...

// Generate a random double RDD that contains 1 million i.i.d. values drawn from the
// standard normal distribution `N(0, 1)`, evenly distributed in 10 partitions.
//  生成1000000个服从正态分配N(0,1)的RDD[Double]，并且分布在 10 个分区中
val u = normalRDD(sc, 1000000L, 10)
// Apply a transform to get a random double RDD following `N(1, 4)`.
// 把生成的随机数转化成N(1,4) 正态分布
val v = u.map(x => 1.0 + 2.0 * x)
v.take(5).foreach(println)
```
console
```text
2.3773282636500435
-3.1977523223033204
3.317517817879228
2.5317479436385444
2.6526840335735447
```
以上为官网示例

以下为bin包里的示例

```scala
package org.apache.spark.examples.mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.rdd.RDD

object RandomRDDGeneration {

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName(s"RandomRDDGeneration").setMaster("local")
    val sc = new SparkContext(conf)

    val numExamples = 10000 // number of examples to generate
    val fraction = 0.1 // fraction of data to sample

    // Example: RandomRDDs.normalRDD
    val normalRDD: RDD[Double] = RandomRDDs.normalRDD(sc, numExamples)
    println(s"Generated RDD of ${normalRDD.count()}" +
      " examples sampled from the standard normal distribution")
    println("  First 5 samples:")
    normalRDD.take(5).foreach( x => println(s"    $x") )

    // Example: RandomRDDs.normalVectorRDD
    val normalVectorRDD = RandomRDDs.normalVectorRDD(sc, numRows = numExamples, numCols = 2)
    println(s"Generated RDD of ${normalVectorRDD.count()} examples of length-2 vectors.")
    println("  First 5 samples:")
    normalVectorRDD.take(5).foreach( x => println(s"    $x") )

    println()

    sc.stop()
  }

}
```
console
```text
Generated RDD of 10000 examples sampled from the standard normal distribution
  First 5 samples:
    0.8241839639816105
    0.5020887737792052
    0.3698313500859245
    -0.15760773859684682
    0.7785335989621256
    
Generated RDD of 10000 examples of length-2 vectors.
  First 5 samples:
    [-0.5651007485027268,-0.06351614534720305]
    [-0.40678775271319767,-0.7803638982689272]
    [-0.8649392982696187,-2.0269659124926185]
    [-1.007061750302515,-0.05030556840147083]
    [-0.7738875459250691,0.41244911335964196]
```







