package org.apache.spark.ml.made

import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.param.{DoubleParam, IntParam}
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql.types.StructType

trait LinearRegressionParams extends HasInputCol with HasOutputCol with HasLabelCol {
  private val learningRate = new DoubleParam(this, "learningRate", "learning rate for GD")
  setDefault(learningRate -> 1.0)
  
  private val maxEpochs = new IntParam(this, "maxEpochs", "maximum epochs for GD")
  setDefault(maxEpochs -> 100)

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def getLearningRate: Double = $(learningRate)

  def setLearningRate(value: Double): this.type = set(learningRate, value)

  def getMaxEpochs: Int = $(maxEpochs)

  def setMaxEpochs(value: Int): this.type = set(maxEpochs, value)

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
