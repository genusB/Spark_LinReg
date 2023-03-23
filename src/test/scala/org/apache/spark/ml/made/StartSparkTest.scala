package org.apache.spark.ml.made

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class StartSparkTest extends AnyFlatSpec with Matchers with WithSpark {

  "Spark" should "start context" in {
    val s = spark
    s.conf.getAll.foreach(u => println(u))

    Thread.sleep(1000)
  }
}
