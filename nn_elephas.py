# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ixxvvfPKUa7_WSwHaF8wqlv8-PA5xKt6
"""

# ! pip install pyspark tensorflow elephas

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from keras.models import Sequential
from keras.layers import Dense
from elephas.spark_model import SparkModel
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
# Initialize Spark session
spark = SparkSession.builder.appName("DistributedDeepLearningRegression").getOrCreate()

# Load data into Spark DataFrame
df = spark.read.parquet("gs://sk777/preprocessed_data_onehot.parquet")

df = df.sample(fraction=0.5, seed=22)
df.show()
row_count = df.count()

print(f"Number of rows in DataFrame: {row_count}")


# In[21]:


# Combine all one-hot encoded features into a single feature vector
assembler = VectorAssembler(
    inputCols=[f"{col}_vec" for col in ['os', 'instance_type', 'region']],
    outputCol="features"
)
df = assembler.transform(df)
df = df.select("features", "price")  # Select only features and target


# In[22]:


from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import Pipeline

# Assemble the target variable into a vector for scaling
target_assembler = VectorAssembler(inputCols=["price"], outputCol="price_vec")

# Scale the target variable
scaler = MinMaxScaler(inputCol="price_vec", outputCol="price_scaled")

# Create a pipeline for target preprocessing
pipeline = Pipeline(stages=[target_assembler, scaler])
df = pipeline.fit(df).transform(df)

# Select final columns
df = df.select("features", "price_scaled")
df = df.withColumnRenamed("price_scaled", "label")

df.show(5)

train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

train_data = train_df.rdd.map(lambda row: (row["features"].toArray(), row["label"]))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from elephas.spark_model import SparkModel
from tensorflow.keras.optimizers.legacy import Adam
# Define the model
def create_model(input_dim):
    model = Sequential([
        Dense(512, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # For regression tasks
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Wrap the model with Elephas
input_dim = len(train_df.first()["features"])

keras_model = create_model(input_dim)
#
### Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

spark_model = SparkModel(model=keras_model, mode='asynchronous', num_workers=2)

# Fit the model with early stopping
spark_model.fit(train_data, epochs=50, batch_size=4096, verbose=1, validation_split=0.1, callbacks=[early_stopping])


# Prepare features RDD for prediction
features_rdd = test_df.rdd.map(lambda row: row["features"].toArray())
predictions = spark_model.predict(features_rdd)

num_partitions = test_df.rdd.getNumPartitions()

# # Convert predictions list to RDD with the same number of partitions as test_df
predictions_rdd = spark.sparkContext.parallelize(predictions, num_partitions)

# Ensure both RDDs have the same partitioning
predictions_rdd = predictions_rdd.repartition(test_df.rdd.getNumPartitions())

from pyspark.sql import Row

# Convert predictions to a format Spark can understand
predictions_rdd = predictions_rdd.map(lambda x: Row(prediction=float(x[0])))

# Convert RDD to DataFrame
predictions_df = predictions_rdd.toDF()

test_df = test_df.repartition(200)  # Adjust the number of partitions as needed
predictions_df = predictions_df.repartition(200)

# Add an index to both DataFrames to align rows
from pyspark.sql.functions import monotonically_increasing_id

test_df = test_df.withColumn("id", monotonically_increasing_id())
predictions_df = predictions_df.withColumn("id", monotonically_increasing_id())

# Join on the "id" column
result_df = test_df.join(predictions_df, on="id").drop("id")

# Now `result_df` contains "label" and "prediction" columns
result_df.show(5)

from pyspark.sql.functions import udf,col
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import DenseVector

# UDF to extract the first element of a sparse vector
def extract_value(vector):
    if vector and hasattr(vector, "values"):
        return float(vector.values[0])
    else:
        return float(vector)

extract_value_udf = udf(extract_value, DoubleType())

# Apply UDF to the label column
result_df = result_df.withColumn("label", extract_value_udf(col("label")))

# Calculate MSE and RMSE
from pyspark.sql.functions import pow, mean

mse = result_df.select(pow(col("label") - col("prediction"), 2).alias("squared_error")) \
               .agg(mean("squared_error").alias("mse")) \
               .collect()[0]["mse"]

rmse = mse ** 0.5

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

spark.stop()