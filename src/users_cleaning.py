"""
users_cleaning.py
Users Data Cleaning and Export Script
Stack Overflow Big Data Project - O1 Objective

This script performs schema normalization + light cleanup on the Users.xml
and exports it to Parquet format.

"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, year, trim, lower, when, nullif, lit
)
from pyspark.sql.types import LongType, IntegerType, DateType


def create_spark_session():
    return (
        SparkSession.builder
        .appName("Users Cleaning and Export")
        .master("local[*]")
        .config("spark.driver.memory", "12g")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )


def load_users_xml(spark, file_path):
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


def clean_users_data(df):
    """
    Perform schema normalization and light cleanup suitable for O1:
    - standardize column names and types
    - trim strings
    - convert empty strings to null
    - null out obvious placeholder website URLs
    - add is_system_user flag
    - add creation_year for partitioning
    """

    initial_count = df.count()
    print(f"Initial record count: {initial_count:,}")

    def clean_str(c):
        return nullif(trim(c), lit(""))

    website_raw = clean_str(col("_WebsiteUrl"))
    
    website_clean = when(
        lower(website_raw).isin("http://none", "https://none", "none", "null", "n/a", "na"),
        None
    ).otherwise(website_raw)

    df_clean = df.select(
        col("_Id").cast(LongType()).alias("user_id"),
        col("_AccountId").cast(LongType()).alias("account_id"),

        clean_str(col("_DisplayName")).alias("display_name"),

        col("_CreationDate").cast(DateType()).alias("creation_date"),
        col("_LastAccessDate").cast(DateType()).alias("last_access_date"),

        col("_Reputation").cast(IntegerType()).alias("reputation"),
        col("_UpVotes").cast(IntegerType()).alias("up_votes"),
        col("_DownVotes").cast(IntegerType()).alias("down_votes"),
        col("_Views").cast(IntegerType()).alias("views"),

        clean_str(col("_Location")).alias("location"),

        website_clean.alias("website_url"),

        clean_str(col("_AboutMe")).alias("about_me"),
    )

    df_clean = df_clean.withColumn("is_system_user", col("user_id") < 0)
    df_clean = df_clean.withColumn("creation_year", year(col("creation_date")))
    df_clean.printSchema()

    print(f"Null user_id: {df_clean.filter(col('user_id').isNull()).count():,}")
    print(f"Null creation_date: {df_clean.filter(col('creation_date').isNull()).count():,}")
    
    print(f"location is null (includes empty->null): {df_clean.filter(col('location').isNull()).count():,}")
    print(f"website_url is null (includes empty/placeholder->null): {df_clean.filter(col('website_url').isNull()).count():,}")
    
    sys_users = df_clean.filter(col("is_system_user") == True).count()
    print(f"System/collective users (user_id < 0): {sys_users:,}")

    print("\nYear distribution (sample):")
    df_clean.groupBy("creation_year").count().orderBy("creation_year").show(20)

    final_count = df_clean.count()
    print(f"\nFinal record count: {final_count:,}")

    return df_clean


def export_to_parquet(df, output_path, coalesce_n=16):
    """
    Export cleaned DataFrame to Parquet with partitioning.
    """
    print_section("STEP 3: EXPORT TO PARQUET")
    print(f"Output path: {output_path}")
    print(f"Coalescing to {coalesce_n} files...")

    (
        df
        .coalesce(coalesce_n)
        .write
        .mode("overwrite")
        .partitionBy("creation_year")
        .option("compression", "snappy")
        .parquet(output_path)
    )

    print("Parquet export complete!")


def validate_parquet(spark, output_path):
    """Validate Parquet output by reading it back."""
    print_section("STEP 4: VALIDATION")

    df_parquet = spark.read.parquet(output_path)

    print(f"Parquet record count: {df_parquet.count():,}")
    print("Parquet schema:")
    df_parquet.printSchema()

    print("Partition distribution:")
    df_parquet.groupBy("creation_year").count().orderBy("creation_year").show(20)


def main():
    print("=" * 60)
    print(" USERS DATA CLEANING & EXPORT - O1 ")
    print("=" * 60)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    input_path = os.path.join(DATA_DIR, "Users.xml")
    output_path = os.path.join(DATA_DIR, "processed", "users")

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print(f"\nLoading data from: {input_path}")
    df = load_users_xml(spark, input_path).cache()
    df.count() 

    df_clean = clean_users_data(df)

    export_to_parquet(df_clean, output_path, coalesce_n=16)

    validate_parquet(spark, output_path)

    print_section("EXPORT COMPLETE")
    spark.stop()


if __name__ == "__main__":
    main()