### Streaming Significance Testing 流式显著性检验
---
`spark.mllib`提供一些测试的在线实现来支持`A/B testing`等场景。这些测试(tests)可以运行在`Spark Streaming DStream[(Boolean, Double)]`上，每个`tuple`的第一个元素表示控制组`(control group(false))`或处理组`(treatment group(true))`,第二个元素是观察值(`the value of an observation`).

流显著性测试支持以下参数:
- peacePeriod: 忽视流中初始数据点的数量，用于减少新奇影响(`mitigate novelty effects`)
- windowSize: 执行假设检验的过往批次的数量.设为0将对所有以往批次做累积处理

```text
peacePeriod - The number of initial data points from the stream to ignore, used to mitigate novelty effects.
windowSize - The number of past batches to perform hypothesis testing over. Setting to 0 will perform cumulative processing using all prior batches.
```
`StreamingTest`提供流式假设检验
```scala
import org.apache.spark.SparkConf
import org.apache.spark.mllib.stat.test.{BinarySample, StreamingTest}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.util.Utils

object StreamingTestExample {

  def main(args: Array[String]) {
    if (args.length != 3) {
      // scalastyle:off println
      System.err.println(
        "Usage: StreamingTestExample " +
          "<dataDir> <batchDuration> <numBatchesTimeout>")
      // scalastyle:on println
      System.exit(1)
    }
    val dataDir = args(0)
    val batchDuration = Seconds(args(1).toLong)
    val numBatchesTimeout = args(2).toInt

    val conf = new SparkConf().setMaster("local").setAppName("StreamingTestExample")
    val ssc = new StreamingContext(conf, batchDuration)
    ssc.checkpoint {
      val dir = Utils.createTempDir()
      dir.toString
    }

    /**
      * Create a temporary directory inside the given parent directory. The directory will be
      * automatically deleted when the VM shuts down.
      */
    val data = ssc.textFileStream(dataDir).map(line => line.split(",") match {
      case Array(label, value) => BinarySample(label.toBoolean, value.toDouble)
    })

    val streamingTest = new StreamingTest()
      .setPeacePeriod(0)
      .setWindowSize(0)
      .setTestMethod("welch")

    val out = streamingTest.registerStream(data)
    out.print()
    // $example off$

    // Stop processing if test becomes significant or we time out
    var timeoutCounter = numBatchesTimeout
    out.foreachRDD { rdd =>
      timeoutCounter -= 1
      val anySignificant = rdd.map(_.pValue < 0.05).fold(false)(_ || _)
      if (timeoutCounter == 0 || anySignificant) rdd.context.stop()
    }

    ssc.start()
    ssc.awaitTermination()
  }
}
```

```text
bin/run-example mllib.StreamingTestExample hdfs://localhost:9000/opt/ 5 100

-------------------------------------------
Time: 1539932455000 ms
-------------------------------------------
Streaming test summary:
method: Welch's 2-sample t-test
degrees of freedom = 3.4494998889665234
statistic = -0.27943239726950414
pValue = 0.7943369776946448
No presumption against null hypothesis: Both groups have same mean.

-------------------------------------------
Time: 1539932460000 ms
-------------------------------------------
Streaming test summary:
method: Welch's 2-sample t-test
degrees of freedom = 3.4494998889665234
statistic = -0.27943239726950414
pValue = 0.7943369776946448
No presumption against null hypothesis: Both groups have same mean.

-------------------------------------------
Time: 1539932465000 ms
-------------------------------------------
Streaming test summary:
method: Welch's 2-sample t-test
degrees of freedom = 3.4494998889665234
statistic = -0.27943239726950414
pValue = 0.7943369776946448
No presumption against null hypothesis: Both groups have same mean.

-------------------------------------------
Time: 1539932470000 ms
-------------------------------------------
Streaming test summary:
method: Welch's 2-sample t-test
degrees of freedom = 3.4494998889665234
statistic = -0.27943239726950414
pValue = 0.7943369776946448
No presumption against null hypothesis: Both groups have same mean.
```
---
### 参考
[1] [显著性检验](https://wiki.mbalib.com/wiki/Significance_Testing)
[2] [Streaming Significance Testing](http://spark.apache.org/docs/latest/mllib-statistics.html#streaming-significance-testing)