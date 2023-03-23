package org.apache.spark.ml.made

import breeze.linalg.sum
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.linalg.Vectors.fromBreeze
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import breeze.linalg.{DenseVector => BreezeDV}

trait LinearRegressionParams
  extends HasInputCol with HasOutputCol with HasLabelCol {
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)

  private val learningRate = new DoubleParam(this, "learningRate", "learning rate for GD")
  def getLearningRate: Double = $(learningRate)
  def setLearningRate(value: Double): this.type = set(learningRate, value)
  setDefault(learningRate -> 1.0)

  private val maxEpochs = new IntParam(this, "maxEpochs", "maximum epochs for GD")
  def getMaxEpochs: Int = $(maxEpochs)
  def setMaxEpochs(value: Int): this.type = set(maxEpochs, value)
  setDefault(maxEpochs -> 100)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression(override val uid: String)
  extends Estimator[LinearRegressionModel] with LinearRegressionParams with DefaultParamsWritable {

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

    for(epoch <- 0 until getMaxEpochs) {
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
    }

    copyValues(
      new LinearRegressionModel(
        Vectors.fromBreeze(w(0 until w.size - 1)).toDense,
        w(w.size - 1)
      )
    ).setParent(this)
}

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](override val uid: String,
                                         val w: DenseVector,
                                         val b: Double)
  extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {


  private[made] def this(w: DenseVector, b: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), w.toDense, b)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(w, b), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x: Vector) => {
        Vectors.dense((x.asBreeze dot w.asBreeze) + b)
      })


    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors: (Vector, Vector) = w.asInstanceOf[Vector] -> Vectors.dense(b)

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val (w, b) = vectors.select(vectors("_1").as[Vector], vectors("_2").as[Vector]).first()

      val model = new LinearRegressionModel(w.toDense, b(0))
      metadata.getAndSetParams(model)
      model
    }
  }
}
