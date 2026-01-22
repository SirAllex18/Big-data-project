"""
Votes Data Profiling Script
Stack Overflow Big Data Project - O1 Objective

Examine the set to see what we are working with and if changes are needed.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, min, max


def create_spark_session():
    return (
        SparkSession.builder
        .appName("Votes Data Profiling")
        .master("local[*]")
        .config("spark.driver.memory", "12g")
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


def profile_votes(df):
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

    # 4. Duplicate Vote IDs
    dup = df.groupBy("_Id").count().filter("count > 1")
    print(f"Duplicate Vote IDs: {dup.count():,}")
    dup.show(5)

    # 5. Vote Type Distribution
    df.groupBy("_VoteTypeId").count().orderBy(col("count").desc()).show()

    invalid_vote_types = df.filter(col("_VoteTypeId") <= 0).count()
    print(f"Invalid VoteTypeId (<=0): {invalid_vote_types:,}")

    # 6. Post ID Analysis
    null_posts = df.filter(col("_PostId").isNull()).count()
    print(f"Votes with null PostId: {null_posts:,}")

    # 7. Date Range
    df.select(
        min("_CreationDate").alias("min_date"),
        max("_CreationDate").alias("max_date")
    ).show(truncate=False)

    # Summary
    print(f"""
    Total Records: {total_rows:,}
    Duplicate Vote IDs: {dup.count():,}
    Invalid Vote Types: {invalid_vote_types:,}
    Null Post IDs: {null_posts:,}
    """)


def main():
    spark = create_spark_session()
    input_path = "../data/Votes.xml"
    df = load_votes_xml(spark, input_path)
    df.cache()
    df.count()

    profile_votes(df)
    spark.stop()


if __name__ == "__main__":
    main()
