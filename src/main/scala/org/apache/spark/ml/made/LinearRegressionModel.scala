package org.apache.spark.ml.made

import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType

class LinearRegressionModel private[made](override val uid: String, val w: DenseVector, val b: Double)
  extends Model[LinearRegressionModel]
    with LinearRegressionParams
    with MLWritable {

  private[made] def this(w: DenseVector, b: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), w.toDense, b)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(w, b), extra)

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

