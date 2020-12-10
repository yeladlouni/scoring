import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassifier, LogisticRegression, NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.sql.SparkSession

object Scoring extends App {
  val spark = SparkSession
    .builder()
    .config("AppName", "Scoring")
    .config("spark.master", "local")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  val df_train = spark.read
    .option("header", true)
    .option("inferSchema", true)
    .csv("C:\\Users\\yelad\\IdeaProjects\\logistic-regression\\data\\train.csv")

  val df_validation = spark.read
    .option("header", true)
    .option("inferSchema", true)
    .csv("C:\\Users\\yelad\\IdeaProjects\\logistic-regression\\data\\validation.csv")

  val df_test = spark.read
    .option("header", true)
    .option("inferSchema", true)
    .csv("C:\\Users\\yelad\\IdeaProjects\\logistic-regression\\data\\test.csv")


  var houseIndexer = new StringIndexer()
    .setInputCol("has_house")
    .setOutputCol("has_house_indexed")

  val statusIndexer = new StringIndexer()
    .setInputCol("marital_status")
    .setOutputCol("marital_status_indexed")

  val creditIndexer = new StringIndexer()
    .setInputCol("credit")
    .setOutputCol("label")

  val assembler = new VectorAssembler()
    .setInputCols(Array("age"))
    .setOutputCol("features")
  val lrModel = new LogisticRegression()
  val rfModel = new RandomForestClassifier()
  val gbtModel = new GBTClassifier()
  val nbModel = new NaiveBayes()

  val pipeline = new Pipeline()
    .setStages(Array(houseIndexer, statusIndexer, creditIndexer, assembler, rfModel))

  val model = pipeline.fit(df_train)

  val predictions = model.transform(df_validation)

  predictions.show()

  val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction")

  val accuracy = evaluator.evaluate(predictions)

  println(accuracy)


}