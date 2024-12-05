from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql.functions import col

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("ModelTraining") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Load the preprocessed data
input_path = "gs://termproject777/preprocessed_data_onehot.parquet"
combined_df = spark.read.parquet(input_path)

# Use OneHotEncoded features directly
feature_cols = ["instance_type_vec", "os_vec", "region_vec"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

# Assemble features and select final dataset
final_df = assembler.transform(combined_df).select("features", col("price").alias("label"))

# Split the data into training and test sets
train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)

# Persist train_df and test_df for better performance
train_df.cache()
test_df.cache()

# Initialize regression models
lr = LinearRegression(featuresCol="features", labelCol="label")
dt = DecisionTreeRegressor(featuresCol="features", labelCol="label")
rf = RandomForestRegressor(featuresCol="features", labelCol="label")

# Set up hyperparameter grids
lr_param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
    .build()

dt_param_grid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10]) \
    .addGrid(dt.maxBins, [32, 64]) \
    .build()

rf_param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [20, 50]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .addGrid(rf.maxBins, [32, 64]) \
    .build()

# Define evaluator
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

# Function for hyperparameter tuning
def hyperparameter_tuning(model, param_grid, train_data, model_name):
    train_val_split = TrainValidationSplit(
        estimator=model,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        trainRatio=0.8
    )
    cv_model = train_val_split.fit(train_data)
    best_model = cv_model.bestModel
    print(f"\nBest Model Parameters for {model_name}:")
    print(best_model.extractParamMap())
    return best_model

# Hyperparameter tuning
print("\nTuning Linear Regression...")
best_lr_model = hyperparameter_tuning(lr, lr_param_grid, train_df, "Linear Regression")

print("\nTuning Decision Tree Regressor...")
best_dt_model = hyperparameter_tuning(dt, dt_param_grid, train_df, "Decision Tree Regressor")

print("\nTuning Random Forest Regressor...")
best_rf_model = hyperparameter_tuning(rf, rf_param_grid, train_df, "Random Forest Regressor")

# Evaluate models
def evaluate_model(model, test_data, model_name):
    predictions = model.transform(test_data)
    rmse = evaluator.evaluate(predictions)
    r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)
    print(f"{model_name} RMSE: {rmse:.4f}, R2: {r2:.4f}")

print("\nModel Performance:")
evaluate_model(best_lr_model, test_df, "Linear Regression")
evaluate_model(best_dt_model, test_df, "Decision Tree Regressor")
evaluate_model(best_rf_model, test_df, "Random Forest Regressor")

# Unpersist DataFrames
train_df.unpersist()
test_df.unpersist()

# Stop the SparkSession
spark.stop()
