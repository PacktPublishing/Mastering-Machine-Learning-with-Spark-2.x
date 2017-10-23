package com.packtpub.mmlwspark.chapter8

import java.io.File

import hex.tree.gbm.GBMModel
import org.apache.spark.h2o.{H2OContext, H2OFrame}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{StructField, StructType}

/**
  * Mastering Machine Learning With Spark: Chapter 8
  *
  * Shared library
  */
object Chapter8Library {
  // @Snippet
  import org.apache.spark.sql.functions._
  val toNumericRate = (rate: String) => {
    val num = if (rate != null) rate.stripSuffix("%").trim else ""
    if (!num.isEmpty) num.toFloat else Float.NaN
  }
  val toNumericRateUdf = udf(toNumericRate)

  // @Snippet
  val toNumericMnths = (replacementValue: Float) => (mnths: String) => {
      if (mnths != null && !mnths.trim.isEmpty) mnths.trim.toFloat else replacementValue
  }
  val toNumericMnthsUdf = udf(toNumericMnths(0.0f))

  // @Snippet
  val toBinaryLoanStatus = (status: String) => status.trim.toLowerCase() match {
      case "fully paid" => "good loan"
      case _ => "bad loan"
  }
  val toBinaryLoanStatusUdf = udf(toBinaryLoanStatus)

  // @Snippet
  def basicDataCleanup(loanDf: DataFrame, colsToDrop: Seq[String] = Seq()) = {
    (
      (if (loanDf.columns.contains("int_rate"))
        loanDf.withColumn("int_rate", toNumericRateUdf(col("int_rate")))
      else
        loanDf)
        .withColumn("revol_util", toNumericRateUdf(col("revol_util")))
        .withColumn("mo_sin_old_il_acct", toNumericMnthsUdf(col("mo_sin_old_il_acct")))
        .withColumn("mths_since_last_delinq", toNumericMnthsUdf(col("mths_since_last_delinq")))
        .withColumn("mths_since_last_record", toNumericMnthsUdf(col("mths_since_last_record")))
        .withColumn("mths_since_last_major_derog", toNumericMnthsUdf(col("mths_since_last_major_derog")))
        .withColumn("mths_since_recent_bc", toNumericMnthsUdf(col("mths_since_recent_bc")))
        .withColumn("mths_since_recent_bc_dlq", toNumericMnthsUdf(col("mths_since_recent_bc_dlq")))
        .withColumn("mths_since_recent_inq", toNumericMnthsUdf(col("mths_since_recent_inq")))
        .withColumn("mths_since_recent_revol_delinq", toNumericMnthsUdf(col("mths_since_recent_revol_delinq")))
    ).drop(colsToDrop.toArray :_*)
  }

  // @Snippet
  val unifyTextColumn = (in: String) => {
      if (in != null) in.toLowerCase.replaceAll("[^\\w ]", " ") else null
    }
  val unifyTextColumnUdf = udf(unifyTextColumn)

  // @Snippet
  val descWordEncoder = (denominatingWords: Array[String]) => (desc: String) => {
      if (desc != null) {
        val unifiedDesc = unifyTextColumn(desc)
        Vectors.dense(denominatingWords.map(w => if (unifiedDesc.contains(w)) 1.0 else 0.0))
      } else null
    }

  val descWordEncoderUdf = udf(descWordEncoder(Array("rate", "interest", "years", "job", "lending")))

  // @Snippet
  def profitMoneyLoss = (predThreshold: Double) =>
      (act: String, predGoodLoanProb: Double, loanAmount: Int, intRate: Double, term: String) => {
        val termInMonths = term.trim match {
          case "36 months" => 36
          case "60 months" => 60
        }
        val intRatePerMonth = intRate / 12 / 100
        if (predGoodLoanProb < predThreshold && act == "good loan") {
          termInMonths*loanAmount*intRatePerMonth / (1 - Math.pow(1+intRatePerMonth, -termInMonths)) - loanAmount
        } else 0.0
  }

  // @Snippet
  val loanMoneyLoss = (predThreshold: Double) =>
      (act: String, predGoodLoanProb: Double, loanAmount: Int) => {
        if (predGoodLoanProb > predThreshold && act == "bad loan") loanAmount.toDouble else 0.0
  }

  // @Snippet
  def totalLoss(actPredDf: DataFrame, threshold: Double): (Double, Double, Long, Double, Long, Double) = {
      import org.apache.spark.sql.Row

      val profitMoneyLossUdf = udf(profitMoneyLoss(threshold))
      val loanMoneyLossUdf = udf(loanMoneyLoss(threshold))

      val lostMoneyDf = actPredDf
        .where("loan_status is not null and loan_amnt is not null")
        .withColumn("profitMoneyLoss", profitMoneyLossUdf(col("loan_status"), col("good loan"), col("loan_amnt"), col("int_rate"), col("term")))
        .withColumn("loanMoneyLoss", loanMoneyLossUdf(col("loan_status"), col("good loan"), col("loan_amnt")))

      lostMoneyDf
        .agg("profitMoneyLoss" -> "sum", "loanMoneyLoss" -> "sum")
        .collect.apply(0) match {
        case Row(profitMoneyLossSum: Double, loanMoneyLossSum: Double) =>
          (threshold,
            profitMoneyLossSum, lostMoneyDf.where("profitMoneyLoss > 0").count,
            loanMoneyLossSum, lostMoneyDf.where("loanMoneyLoss > 0").count,
            profitMoneyLossSum + loanMoneyLossSum
          )
      }
  }

  // @Snippet
  def toHf(df: DataFrame, name: String)(h2oContext: H2OContext): H2OFrame = {
      val hf = h2oContext.asH2OFrame(df, name)
      val allStringColumns = hf.names().filter(name => hf.vec(name).isString)
      hf.colToEnum(allStringColumns)
      hf
  }

  // @Snippet
  import _root_.hex.tree.drf.DRFModel
  def scoreLoan(df: DataFrame,
                      empTitleTransformer: PipelineModel,
                      loanStatusModel: DRFModel,
                      goodLoanProbThreshold: Double,
                      intRateModel: GBMModel)(h2oContext: H2OContext): DataFrame = {
    val inputDf = empTitleTransformer.transform(basicDataCleanup(df))
      .withColumn("desc_denominating_words", descWordEncoderUdf(col("desc")))
      .drop("desc")
    val inputHf = toHf(inputDf, "input_df_" + df.hashCode())(h2oContext)
    // Predict loan status and int rate
    val loanStatusPrediction = loanStatusModel.score(inputHf)
    val intRatePrediction = intRateModel.score(inputHf)
    val probGoodLoanColName = "good loan"
    val inputAndPredictionsHf = loanStatusPrediction.add(intRatePrediction).add(inputHf)
    inputAndPredictionsHf.update()
    // Prepare field loan_status based on threshold
    val loanStatus = (threshold: Double) => (predGoodLoanProb: Double) => if (predGoodLoanProb < threshold) "bad loan" else "good loan"
    val loanStatusUdf = udf(loanStatus(goodLoanProbThreshold))
    h2oContext.asDataFrame(inputAndPredictionsHf)(df.sqlContext).withColumn("loan_status", loanStatusUdf(col(probGoodLoanColName)))
  }

    def saveSchema(schema: StructType, destFile: File, saveWithMetadata: Boolean = false) = {
      import java.nio.file.{Files, Paths, StandardOpenOption}

      import org.apache.spark.sql.types._
      val processedSchema = StructType(schema.map {
        case StructField(name, dtype, nullable, metadata) => StructField(name, dtype, nullable, if (saveWithMetadata) metadata else Metadata.empty)
        case rec => rec
       })

      Files.write(Paths.get(destFile.toURI),
                  processedSchema.json.getBytes(java.nio.charset.StandardCharsets.UTF_8),
                  StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)
    }

  def loadSchema(srcFile: File): StructType = {
    import org.apache.spark.sql.types.DataType
    StructType(
      DataType.fromJson(scala.io.Source.fromFile(srcFile).mkString).asInstanceOf[StructType].map {
        case StructField(name, dtype, nullable, metadata) => StructField(name, dtype, true, metadata)
        case rec => rec
      }
    )
  }
}
