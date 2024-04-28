import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Start the Spark session
spark = SparkSession.builder \
    .appName('WineQuality') \
    .getOrCreate()

# Load the dataset
df = spark.read.csv("TrainingDataset.csv", header=True, sep=";", inferSchema=True)

# Remove quotes from column names and rename 'quality' to 'label'
df = df.toDF(*(c.strip('"') for c in df.columns))
df = df.withColumnRenamed('quality', 'label')

# Ensure all columns are of type float
df = df.select([df[col].cast('float').alias(col) for col in df.columns])

# Print modified schema to check column names and types
df.printSchema()

# Create feature vector
assembler = VectorAssembler(inputCols=[col for col in df.columns if col != 'label'], outputCol='features')
final_data = assembler.transform(df).select('features', 'label')

# Split the data into training and test sets
train_data, test_data = final_data.randomSplit([0.7, 0.3])

# Create a RandomForest model
rf = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=20)

# Fit the model
model = rf.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Save the model
model.write().overwrite().save('s3://winequal/trainingmodel.model')

# Stop the Spark session
spark.stop()
