package com.packtpub.mmlwspark.utils

import org.apache.spark.h2o.H2OContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.expressions.UserDefinedFunction
import water.fvec.H2OFrame

/**
  * Book utils.
  */
object Utils {
  def colTransform(hf: H2OFrame, udf: UserDefinedFunction, colName: String)(implicit h2oContext: H2OContext, sqlContext: SQLContext): H2OFrame = {
    import sqlContext.implicits._
    val name = hf.key.toString
    val colHf = hf(Array(colName))
    val df = h2oContext.asDataFrame(colHf)
    val result = h2oContext.asH2OFrame(df.withColumn(colName, udf($"${colName}")), s"${name}_${colName}")
    colHf.delete()
    result
  }

  def let[A](in: A)(body: A => Unit) = {
    body(in)
    in
  }
}
