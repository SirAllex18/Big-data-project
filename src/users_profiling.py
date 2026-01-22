"""
Users Data Profiling Script
Stack Overflow Big Data Project - O1 Objective

This script profiles the Users.xml dataset to assess data quality,
schema consistency, and potential cleaning requirements.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, min, max, trim, lower, length


def create_spark_session():
    return (
        SparkSession.builder
        .appName("Users Data Profiling")
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


def profile_users(df):
    # 1. Schema and Row Count
    df.printSchema()
    total_rows = df.count()
    print(f"\nTotal Rows: {total_rows:,}")

    # 2. Sample Data
    df.show(10, truncate=False)

    # 3. Null Analysis
    nulls = df.select([
        count(when(col(c).isNull(), c)).alias(c)
        for c in df.columns
    ])
    nulls.show(truncate=False)

    print("\nNull percentages:")
    for c in df.columns:
        cnt = df.filter(col(c).isNull()).count()
        pct = (cnt / total_rows) * 100
        print(f"  {c}: {cnt:,} ({pct:.4f}%)")

    # 4. Duplicate User IDs
    dup = df.groupBy("_Id").count().filter("count > 1")
    print(f"Duplicate User IDs: {dup.count():,}")
    dup.show(5)

    # 5. Reputation Analysis
    df.select(
        min("_Reputation").alias("min_rep"),
        max("_Reputation").alias("max_rep")
    ).show()

    neg_rep = df.filter(col("_Reputation") < 0).count()
    print(f"Users with negative reputation: {neg_rep:,}")

    # 6. Creation / Last Access Dates
    df.select(
        min("_CreationDate").alias("min_creation"),
        max("_CreationDate").alias("max_creation"),
        min("_LastAccessDate").alias("min_access"),
        max("_LastAccessDate").alias("max_access")
    ).show(truncate=False)

    # 7. Location Analysis
    df.groupBy("_Location").count().orderBy(col("count").desc()).show(10)

    empty_location = df.filter(trim(col("_Location")) == "").count()
    print(f"Empty location strings: {empty_location:,}")

    # 8. Website URL Analysis
    null_urls = df.filter(col("_WebsiteUrl").isNull()).count()
    print(f"Null website URLs: {null_urls:,}")

    df.groupBy("_WebsiteUrl").count().orderBy(col("count").desc()).show(5)

    # 9. Display Name Quality
    empty_names = df.filter(trim(col("_DisplayName")) == "").count()
    print(f"Empty display names: {empty_names:,}")

    whitespace_names = df.filter(
        col("_DisplayName") != trim(col("_DisplayName"))
    ).count()
    print(f"Names with leading/trailing whitespace: {whitespace_names:,}")

    # 10. Data Type Summary
    for field in df.schema:
        print(f"{field.name}: {field.dataType}")

    # Summary
    print(f"""
    Total Records: {total_rows:,}
    Duplicate User IDs: {dup.count():,}
    Negative Reputation Users: {neg_rep:,}
    Empty Display Names: {empty_names:,}
    Empty Locations: {empty_location:,}
    """)


def main():
    spark = create_spark_session()
    input_path = "../data/Users.xml"
    df = load_users_xml(spark, input_path)
    df.cache()
    df.count()

    profile_users(df)
    spark.stop()


if __name__ == "__main__":
    main()
