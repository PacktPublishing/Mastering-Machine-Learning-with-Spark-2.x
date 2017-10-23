package com.packtpub.mmlwspark.chapter8

import java.io.File

import com.packtpub.mmlwspark.chapter8.Chapter8Library.descWordEncoderUdf
import hex.ModelCategory
import hex.genmodel.MojoModel
import hex.genmodel.easy.prediction.{AbstractPrediction, BinomialModelPrediction, RegressionModelPrediction}
import hex.genmodel.easy.{EasyPredictModelWrapper, RowData}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{PipelineModel, Transformer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/**
  * Mastering Machine Learning With Spark: Chapter 8
  *
  * Model deployment
  */
object Chapter8StreamApp extends App {

  val spark = SparkSession.builder()
    .master("local[*]")
    .appName("Chapter8StreamApp")
    .getOrCreate()


  script(spark,
         sys.env.get("MODELSDIR").getOrElse("models"),
         sys.env.get("APPDATADIR").getOrElse("appdata"))

  def script(ssc: SparkSession, modelDir: String, dataDir: String): Unit = {


    // @Snippet
    val inputSchema = Chapter8Library.loadSchema(new File(modelDir, "inputSchema.json"))
    // @Snippet
    val inputDataStream = spark.readStream
      .schema(inputSchema)
      //.option("mode", "DROPMALFORMED")
      .option("timestampFormat", "MMM-yyy")
      .option("nullValue", null)
      .csv(s"${dataDir}/*.csv")
    
    // @Snippet
    inputDataStream.schema.printTreeString()

    // @Snippet
    val empTitleTransformer = PipelineModel.load(s"${modelDir}/empTitleTransformer")
    // @Snippet
    val loanStatusModel = MojoModel.load(new File(s"${modelDir}/loanStatusModel.mojo").getAbsolutePath)
    val intRateModel = MojoModel.load(new File(s"${modelDir}/intRateModel.mojo").getAbsolutePath)

    // @Snippet
    val loanStatusTransformer = new MojoTransformer("loanStatus", loanStatusModel)
    val intRateTransformer = new MojoTransformer("intRate", intRateModel)
    
    // @Snippet
    val outputDataStream =
      intRateTransformer.transform(
        loanStatusTransformer.transform(
          empTitleTransformer.transform(
            Chapter8Library.basicDataCleanup(inputDataStream))
            .withColumn("desc_denominating_words", descWordEncoderUdf(col("desc"))))
    )
    // @Snippet
    outputDataStream.schema.printTreeString()

    outputDataStream.writeStream.format("console").start().awaitTermination()
  }

  def parseArgs(): (String, Double) = {
    val modelsPath = args(0)
    val loanStatusThreshold = args(1).toDouble
    (modelsPath, loanStatusThreshold)
  }
  
  class MojoTransformer(override val uid: String,
                        mojoModel: MojoModel) extends Transformer {

    case class BinomialPrediction(p0: Double, p1: Double)
    case class RegressionPrediction(value: Double)

    implicit def toBinomialPrediction(bmp: AbstractPrediction) = BinomialPrediction(bmp.asInstanceOf[BinomialModelPrediction].classProbabilities(0),
                                                                                    bmp.asInstanceOf[BinomialModelPrediction].classProbabilities(1))
    implicit def toRegressionPrediction(rmp: AbstractPrediction) = RegressionPrediction(rmp.asInstanceOf[RegressionModelPrediction].value)

    val modelUdf = {
      val epmw = new EasyPredictModelWrapper(mojoModel)
      mojoModel._category match {
        case ModelCategory.Binomial => udf[BinomialPrediction, Row] { r: Row => epmw.predict(rowToRowData(r)) }
        case ModelCategory.Regression => udf[RegressionPrediction, Row] { r: Row => epmw.predict(rowToRowData(r)) }
      }
    }

    val predictStruct = mojoModel._category match {
      case ModelCategory.Binomial =>  StructField("p0", DoubleType)::StructField("p1", DoubleType)::Nil
      case ModelCategory.Regression => StructField("pred", DoubleType)::Nil
    }

    val outputCol = s"${uid}Prediction"

    override def transform(dataset: Dataset[_]): DataFrame = {
      val inputSchema = dataset.schema
      val args = inputSchema.fields.map(f => dataset(f.name))
      dataset.select(col("*"), modelUdf(struct(args: _*)).as(outputCol))
    }

    private def rowToRowData(row: Row): RowData = new RowData {
      row.schema.fields.foreach(f => {
        row.getAs[AnyRef](f.name) match {
          case v: Number => put(f.name, v.doubleValue().asInstanceOf[Object])
          case v: java.sql.Timestamp => put(f.name, v.getTime.toDouble.asInstanceOf[Object])
          case null => // nop
          case v => put(f.name, v)
        }
      })
    }
    
    override def copy(extra: ParamMap): Transformer =  defaultCopy(extra)

    override def transformSchema(schema: StructType): StructType =  {
      val outputFields = schema.fields :+ StructField(outputCol, StructType(predictStruct), false)
      StructType(outputFields)
    }
  }
}
