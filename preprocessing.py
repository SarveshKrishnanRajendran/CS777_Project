from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from google.cloud import storage

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("AWSSpotPricePreprocessing") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Set the GCS bucket name
bucket_name = "termproject777"

# Initialize the GCS client
client = storage.Client()

# Get the bucket
bucket = client.get_bucket(bucket_name)

# List all CSV files in the bucket
csv_files = [f"gs://{bucket_name}/{blob.name}" for blob in bucket.list_blobs() if blob.name.endswith(".csv")]
print(f"Found CSV files: {csv_files}")

# Explicitly define the schema
schema = StructType([
    StructField("datetime", StringType(), True),
    StructField("instance_type", StringType(), True),
    StructField("os", StringType(), True),
    StructField("region", StringType(), True),
    StructField("price", FloatType(), True)
])

# Read and combine all CSVs with the predefined schema
combined_df = None
for file in csv_files:
    df = spark.read.csv(file, header=False, schema=schema)
    combined_df = df if combined_df is None else combined_df.union(df)

# Step 1: Remove duplicate rows
combined_df = combined_df.dropDuplicates()

# Step 2: Drop rows with missing values
combined_df = combined_df.dropna()

# Step 3: Filter out invalid or outlier rows (e.g., negative prices or zero prices)
combined_df = combined_df.filter(col("price") > 0)

# Step 4: Encode categorical features using StringIndexer
instance_indexer = StringIndexer(inputCol="instance_type", outputCol="instance_type_index")
os_indexer = StringIndexer(inputCol="os", outputCol="os_index")
region_indexer = StringIndexer(inputCol="region", outputCol="region_index")

combined_df = instance_indexer.fit(combined_df).transform(combined_df)
combined_df = os_indexer.fit(combined_df).transform(combined_df)
combined_df = region_indexer.fit(combined_df).transform(combined_df)

# Step 5: Apply OneHotEncoding to indexed columns
instance_encoder = OneHotEncoder(inputCol="instance_type_index", outputCol="instance_type_vec")
os_encoder = OneHotEncoder(inputCol="os_index", outputCol="os_vec")
region_encoder = OneHotEncoder(inputCol="region_index", outputCol="region_vec")

combined_df = instance_encoder.fit(combined_df).transform(combined_df)
combined_df = os_encoder.fit(combined_df).transform(combined_df)
combined_df = region_encoder.fit(combined_df).transform(combined_df)

# Step 6: Drop unused or original columns
combined_df = combined_df.drop("datetime", "instance_type", "os", "region")

# Save the preprocessed data to GCS as Parquet
output_path = "gs://termproject777/preprocessed_data_onehot.parquet"
combined_df.write.parquet(output_path, mode="overwrite")

print(f"Preprocessed data saved to {output_path}")

# Stop the SparkSession
spark.stop()
