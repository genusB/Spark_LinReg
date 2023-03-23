package org.apache.spark.ml.made

import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec._
import org.scalatest.matchers._
import breeze.linalg.{*, DenseMatrix, DenseVector}
import org.apache.spark.ml.linalg.Vectors.fromBreeze
import org.apache.spark.ml.{Pipeline, PipelineModel}
import com.google.common.io.Files

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.1
  val w: DenseVector[Double] = LinearRegressionTest._w
  val b: Double = LinearRegressionTest._b
  val y: DenseVector[Double] = LinearRegressionTest._y
  val dataframe: DataFrame = LinearRegressionTest._dataframe

  private def validateWeights(linreg: LinearRegressionModel): Unit = {
    linreg.w.size should be (w.size)

    for (i <- 0 until w.size) {
      linreg.w(i) should be(w(i) +- delta)
    }

    linreg.b should be(b +- delta)
  }

  private def validatePredictions(data: DataFrame): Unit = {
    data.collect().length should be(dataframe.collect().length)

    val yPred = data.collect().map(_.getAs[Double](1))

    yPred.length should be(LinearRegressionTest.length)

    for (i <- 0 until yPred.length) {
      yPred(i) should be(y(i) +- delta)
    }
  }

  "Estimator" should "fit model" in {
    val estimator = new LinearRegression()
      .setInputCol("x")
      .setLabelCol("y")
      .setOutputCol("pred")

    val model = estimator.fit(dataframe)

    validateWeights(model)
  }

  "Model" should "make predictions" in {
    val model = new LinearRegressionModel(w=fromBreeze(w).toDense, b=b)
      .setInputCol("x")
      .setLabelCol("y")
      .setOutputCol("pred")

    validatePredictions(model.transform(dataframe))
  }

  "Estimator" should "be able to save and load weights" in {
    val estimator = new LinearRegression()
      .setInputCol("x")
      .setLabelCol("y")
      .setOutputCol("pred")

    val pipeline = new Pipeline()
      .setStages(Array(estimator))

    val tempDir = Files.createTempDir()
    pipeline.write.overwrite().save(tempDir.getAbsolutePath)

    val load = Pipeline.load(tempDir.getAbsolutePath)

    val estimator_load = load.fit(dataframe)
      .stages(0)
      .asInstanceOf[LinearRegressionModel]

    validateWeights(estimator_load)
  }

  "Model" should "be able to save and load weights" in {
    val estimator = new LinearRegression()
      .setInputCol("x")
      .setLabelCol("y")
      .setOutputCol("pred")

    val pipeline = new Pipeline()
      .setStages(Array(estimator))

    val tempDir = Files.createTempDir()

    pipeline.fit(dataframe).write.overwrite().save(tempDir.getAbsolutePath)

    val load = PipelineModel.load(tempDir.getAbsolutePath)

    validatePredictions(load.transform(dataframe))
  }
}

object LinearRegressionTest extends WithSpark {

  import sqlc.implicits._

  val length: Int = 10000
  val numFeatures: Int = 2

  lazy val _x: DenseMatrix[Double] = DenseMatrix.rand[Double](length, numFeatures)
  lazy val _w: DenseVector[Double] = DenseVector(10.1, -2.2)
  lazy val _b: Double = 3.0
  lazy val _y: DenseVector[Double] = _x * _w + _b

  lazy val _matrix = DenseMatrix.horzcat(_x, _y.asDenseMatrix.t)
  lazy val _data = _matrix(*, ::).iterator
    .map(x => Tuple3(x(0), x(1), x(2)))
    .toSeq.toDF("x1", "x2", "y")

  lazy val _assembler = new VectorAssembler()
    .setInputCols(Array("x1", "x2"))
    .setOutputCol("x")

  lazy val _dataframe: DataFrame = _assembler.transform(_data).select("x", "y")
}
