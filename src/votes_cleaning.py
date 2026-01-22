"""
Votes Data Cleaning and Export Script
Stack Overflow Big Data Project - O1 Objective

This script performs minimal, schema-level preparation on the Votes.xml dataset
and exports it to Parquet format.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year
from pyspark.sql.types import LongType, IntegerType, DateType
import os


def create_spark_session():
    return (
        SparkSession.builder
        .appName("Votes Cleaning and Export")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )


def load_votes_xml(spark, file_path):
    return (
        spark.read
        .format("xml")
        .option("rowTag", "row")
        .load(file_path)
    )


def print_section(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def clean_votes_data(df):
    """
    Perform schema normalization and feature derivation on Votes data.
    """

    initial_count = df.count()
    print(f"Initial record count: {initial_count:,}")

    df_clean = df.select(
        col("_Id").cast(LongType()).alias("vote_id"),
        col("_PostId").cast(LongType()).alias("post_id"),
        col("_UserId").cast(LongType()).alias("user_id"),
        col("_VoteTypeId").cast(IntegerType()).alias("vote_type_id"),
        col("_BountyAmount").cast(IntegerType()).alias("bounty_amount"),
        col("_CreationDate").cast(DateType()).alias("vote_date")
    )

    print("Normalized schema:")
    df_clean.printSchema()
    df_clean = df_clean.withColumn("vote_year", year(col("vote_date")))

    print("Added 'vote_year' column for partitioning")
    print("Year distribution (sample):")
    df_clean.groupBy("vote_year").count().orderBy("vote_year").show(20)

    final_count = df_clean.count()
    print(f"Final record count: {final_count:,}")

    return df_clean


def export_to_parquet(df, output_path):
    """
    Export cleaned DataFrame to Parquet with partitioning.
    """

    print_section("STEP 3: EXPORT TO PARQUET")
    print(f"Output path: {output_path}")

    (
        df
        .coalesce(16)
        .write
        .mode("overwrite")
        .partitionBy("vote_year")
        .option("compression", "snappy")
        .parquet(output_path)
    )

    print("Parquet export complete!")


def validate_parquet(spark, output_path):
    """
    Validate Parquet output by reading it back.
    """

    df_parquet = spark.read.parquet(output_path)

    print(f"Parquet record count: {df_parquet.count():,}")
    print("Parquet schema:")
    df_parquet.printSchema()

    print("Partition distribution:")
    df_parquet.groupBy("vote_year").count().orderBy("vote_year").show(20)


def main():
    print("=" * 60)
    print(" VOTES DATA CLEANING & EXPORT - O1 ")
    print("=" * 60)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    input_path = os.path.join(DATA_DIR, "Votes.xml")
    output_path = os.path.join(DATA_DIR, "processed", "votes")

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print(f"Loading data from: {input_path}")
    df = load_votes_xml(spark, input_path).cache()
    df.count() 

    df_clean = clean_votes_data(df)
    export_to_parquet(df_clean, output_path)
    validate_parquet(spark, output_path)

    spark.stop()


if __name__ == "__main__":
    main()