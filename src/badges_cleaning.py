"""
Badges Data Cleaning and Preparation for exporting as Parquet type.

This script performs data cleaning and transformation on the Badges.xml dataset,
then exports it to Parquet format for efficient storage and querying.

Cleaning Steps:
1. Schema standardization (rename columns, optimize types)
2. Data validation (check for anomalies)
3. Feature extraction (add year for partitioning)
4. Export to Parquet with partitioning
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, year, count, when, lit,
    regexp_replace, lower, length
)
from pyspark.sql.types import IntegerType, LongType, StringType, BooleanType, TimestampType
import os


def create_spark_session():
    """Create Spark session with XML support."""
    return SparkSession.builder \
        .appName("Badges Data Cleaning") \
        .master("local[*]") \
        .config("spark.driver.memory", "12g") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()


def load_badges_xml(spark, file_path):
    """Load Badges XML file into DataFrame."""
    return spark.read \
        .format("xml") \
        .option("rowTag", "row") \
        .load(file_path)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def validate_string_anomalies(df, column_name):
    """
    Check for string representations of missing values.
    Returns count of anomalies found.
    Should not be the case since its XML and such a value would return a parsing error of the XML but DEFENSIVE PROGRAMMING is the name of the game
    """
    anomaly_patterns = ["nan", "null", "n/a", "na", "none", "-", ""]

    anomaly_count = df.filter(
        lower(trim(col(column_name))).isin(anomaly_patterns)
    ).count()

    return anomaly_count


def clean_badges_data(df):
    """
    Perform all cleaning transformations on the badges DataFrame.

    Transformations:
    1. Rename columns (remove _ prefix from XML parsing)
    2. Optimize data types (class: Long -> Integer)
    3. Trim whitespace from string fields
    4. Add year column for partitioning
    """

    print_section("STEP 1: Schema adjustments")

    # Get initial count
    initial_count = df.count()
    print(f"Initial record count: {initial_count:,}")

    # Rename columns and optimize types
    df_cleaned = df.select(
        col("_Id").cast(LongType()).alias("id"),
        col("_UserId").cast(LongType()).alias("user_id"),
        lower(trim(col("_Name"))).alias("name"),
        col("_Date").cast(TimestampType()).alias("date"),
        col("_Class").cast(IntegerType()).alias("badge_class"),  # Renamed to avoid SQL reserved word
        col("_TagBased").cast(BooleanType()).alias("tag_based")
    )

    print("\nNew schema:")
    df_cleaned.printSchema()

    print_section("STEP 2: Check for anomalies")

    name_anomalies = validate_string_anomalies(df, "_Name")
    print(f"String anomalies in 'name' column (nan, null, n/a, etc.): {name_anomalies}")

    if name_anomalies > 0:
        print("WARNING: Found string anomalies - these will be kept but flagged")
        df.filter(
            lower(trim(col("_Name"))).isin(["nan", "null", "n/a", "na", "none", "-", ""])
        ).show(5)

    print_section("STEP 3: Add derived column")

    # Add year column for partitioning
    df_cleaned = df_cleaned.withColumn("badge_year", year(col("date")))

    print("\nYear distribution:")
    df_cleaned.groupBy("badge_year").count().orderBy("badge_year").show(20)

    print_section("STEP 4: Summary of clean-up:")

    final_count = df_cleaned.count()
    print(f"Final record count: {final_count:,}")
    print(f"Records removed: {initial_count - final_count:,}")

    # Show sample of cleaned data
    print("\nSample of cleaned data:")
    df_cleaned.show(10, truncate=False)

    # Show final schema
    print("\nFinal schema:")
    df_cleaned.printSchema()

    # Document special cases
    print("\nSpecial cases in data:")
    print(f"  - System user (user_id = -1): 1 record ('Not a Robot' badge)")

    return df_cleaned


def export_to_parquet(df, output_path, partition_by=None):
    """
    Export DataFrame to Parquet format.

    Args:
        df: Cleaned DataFrame
        output_path: Output directory path
        partition_by: Column(s) to partition by (optional)
    """
    print_section("STEP 5: EXPORT TO PARQUET")

    print(f"Output path: {output_path}")

    writer = df.write.mode("overwrite")

    if partition_by:
        print(f"Partitioning by: {partition_by}")
        writer = writer.partitionBy(partition_by)

    writer = writer.option("compression", "snappy")

    print("\nWriting Parquet files...")
    writer.parquet(output_path)

    print("Export complete!")

    return output_path


def validate_parquet_output(spark, output_path):
    """
    Validate the exported Parquet files by reading them back.
    """
    print_section("STEP 6: VALIDATION")

    print(f"Reading back from: {output_path}")
    df_parquet = spark.read.parquet(output_path)

    parquet_count = df_parquet.count()
    print(f"Parquet record count: {parquet_count:,}")

    print("\nParquet schema:")
    df_parquet.printSchema()

    print("\nSample from Parquet:")
    df_parquet.show(5, truncate=False)

    # Show partition structure
    print("\nPartition statistics (by year):")
    df_parquet.groupBy("badge_year").count().orderBy("badge_year").show()

    return parquet_count


def main():
    """Main entry point."""
    print("=" * 60)
    print(" BADGES DATA CLEANING AND PREPARATION")
    print("=" * 60)

    # Configuration
    input_path = "D:/Projects/Big-data-project/data/Badges.xml"
    output_path = "D:/Projects/Big-data-project/data/processed/badges"

    # Initialize Spark
    print("\nInitializing Spark session...")
    spark = create_spark_session()

    print(f"\nLoading data from: {input_path}")
    df = load_badges_xml(spark, input_path)

    # Clean data
    df_cleaned = clean_badges_data(df)

    # Export to Parquet
    export_to_parquet(df_cleaned, output_path, partition_by="badge_year")

    # Validate output
    parquet_count = validate_parquet_output(spark, output_path)

    # Final summary
    print_section("CLEANING COMPLETE")
    print(f"""
    Input: {input_path}
    Output: {output_path}
    Format: Parquet (snappy compression)
    Partitioning: By year (badge_year)

    Records processed: {parquet_count:,}

    Schema changes:
      _Id -> id (LongType)
      _UserId -> user_id (LongType)
      _Name -> name (StringType, trimmed)
      _Date -> date (TimestampType)
      _Class -> badge_class (IntegerType)
      _TagBased -> tag_based (BooleanType)
      [NEW] badge_year (IntegerType, derived from date)
    """)

    spark.stop()


if __name__ == "__main__":
    main()
