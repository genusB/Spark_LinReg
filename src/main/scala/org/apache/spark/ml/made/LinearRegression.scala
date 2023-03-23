package org.apache.spark.ml.made

import breeze.linalg.{sum, DenseVector => BreezeDV}
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.mllib.linalg.Vectors.fromBreeze
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, Encoder}

class LinearRegression(override val uid: String)
  extends Estimator[LinearRegressionModel]
    with LinearRegressionParams
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val datasetTrain: Dataset[_] = dataset.withColumn("b", lit(1))
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array($(inputCol), "b", $(labelCol)))
      .setOutputCol(("train"))

    val vectors = assembler.transform(datasetTrain).select("train").as[Vector]

    val numFeatures = MetadataUtils.getNumFeatures(datasetTrain, $(inputCol))
    var w: BreezeDV[Double] = BreezeDV.rand[Double](numFeatures + 1)

    (0 until getMaxEpochs).foreach(_ => {
      val summary = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
        val result = new MultivariateOnlineSummarizer()
        data.foreach(vector => {
          val x: BreezeDV[Double] = vector.asBreeze(0 until w.size).toDenseVector
          val y = vector.asBreeze(-1)
          val yPred = sum(x * w)
          val grad = x * (yPred - y)
          result.add(fromBreeze(grad))
        })
        Iterator(result)
      }).reduce(_ merge _)

      w = w - getLearningRate * summary.mean.asBreeze
    })

    copyValues(new LinearRegressionModel(Vectors.fromBreeze(w(0 until w.size - 1)).toDense, w(w.size - 1)))
      .setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]
