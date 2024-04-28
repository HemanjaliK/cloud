from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import sys

# Initialize Spark Session
def initialize_spark(app_name='WineQualityPrediction', master='local'):
    spark = SparkSession.builder.appName(app_name).master(master).getOrCreate()
    return spark

# Load data and preprocess
def load_and_preprocess_data(spark, file_path):
    # Load CSV into DataFrame
    df = spark.read.csv(file_path, header=True, inferSchema=True, sep=";")
    df.printSchema()

    # Rename quality to label
    df = df.withColumnRenamed("quality", "label")
    # Convert all columns except 'label' to floats
    for col_name in df.columns[:-1]:
        df = df.withColumn(col_name, df[col_name].cast('float'))
    
    # Assemble features
    assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="features")
    df = assembler.transform(df).select("features", "label")
    return df

# Load or train the model
def load_or_train_model(df, model_path=None):
    if model_path:
        model = RandomForestClassificationModel.load(model_path)
        print("Model loaded successfully")
    else:
        # Training the model
        model = RandomForestClassifier(numTrees=10, featuresCol="features", labelCol="label")
        model = model.fit(df)
    return model

# Evaluate the model
def evaluate_model(model, df):
    predictions = model.transform(df)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1_score = f1_evaluator.evaluate(predictions)
    print("Accuracy = %g" % (accuracy))
    print("F1 Score = %g" % (f1_score))

    # Show detailed report in the form of DataFrame
    predictions.groupBy('label', 'prediction').count().show()

if __name__ == "__main__":
    # Assuming the CSV file path is passed as a command-line argument
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'path_to_default_file.csv'
    model_path = '/winepredict/trainingmodel.model/'  # Set this path to None to train a new model

    spark = initialize_spark()
    df = load_and_preprocess_data(spark, file_path)
    model = load_or_train_model(df, model_path)
    evaluate_model(model, df)
    spark.stop()
