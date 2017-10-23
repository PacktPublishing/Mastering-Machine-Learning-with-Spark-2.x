package org.apache.spark.ml

import java.io._

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.util._
import org.apache.spark.sql.types.DataType

/**
  * Free-style transformer defined based on passed function.
  *
  * This class needs to be in Spark package since it depends
  * on private[ml] part of Spark.
  */
class UDFTransformer[T, U](override val uid: String,
                           f: T => U,
                           inType: DataType,
                           outType: DataType)
  extends UnaryTransformer[T, U, UDFTransformer[T, U]] with MLWritable {

  def this() = this("", null, null, null)

  override protected def createTransformFunc: T => U = f

  override protected def validateInputType(inputType: DataType): Unit = require(inputType == inType)

  override protected def outputDataType: DataType = outType

  override def write: MLWriter =  new UDFWriter(this)
}

object UDFTransformer extends MLReadable[UDFTransformer[_, _]] {

  override def read: MLReader[UDFTransformer[_, _]] = new UDFReader
}

class UDFReader extends MLReader[UDFTransformer[_, _]] {

  override def load(path: String): UDFTransformer[_, _] = {
    val metadata = DefaultParamsReader.loadMetadata(path, sc)
    val modelPath = new Path(path, "model").toString
    val model = sc.objectFile[UDFTransformer[_,_]](modelPath, 1).first()
    model
  }
}

class UDFWriter(instance: UDFTransformer[_, _]) extends MLWriter with Serializable {
  override protected def saveImpl(path: String): Unit = {
    DefaultParamsWriter.saveMetadata(instance, path, sc)
    val modelPath = new Path(path, "model").toString
    sc.parallelize(Seq(instance), 1).saveAsObjectFile(modelPath)
  }
}
