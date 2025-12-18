"""
Badges Data Profiling Script
StackOverflow dataset: Badges table

This script analyzes the Badges.xml dataset to understand what we have in hands :)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, min, max, trim, length, isnan, isnull


def create_spark_session():
    """Create Spark session with XML support."""
    return SparkSession.builder \
        .appName("Badges Data Profiling") \
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


def profile_badges(df):
    """Run all profiling queries on the badges DataFrame."""

    # 1. Schema and Row Count
    print_section("1. SCHEMA AND ROW COUNT")
    print("\nSchema:")
    df.printSchema()

    total_rows = df.count()
    print(f"\nTotal Rows: {total_rows:,}")

    # 2. Sample Data
    print_section("2. SAMPLE DATA (First 10 rows)")
    df.show(10, truncate=False)

    # 3. Null/Missing Value Analysis
    print_section("3. NULL VALUE ANALYSIS")
    null_counts = df.select([
        count(when(col(c).isNull(), c)).alias(c)
        for c in df.columns
    ])
    print("\nNull counts per column:")
    null_counts.show(truncate=False)

    # Calculate percentages
    print("\nNull percentages:")
    for column in df.columns:
        null_count = df.filter(col(column).isNull()).count()
        percentage = (null_count / total_rows) * 100 if total_rows > 0 else 0
        print(f"  {column}: {null_count:,} nulls ({percentage:.4f}%)")

    # 4. Duplicate ID Analysis
    print_section("4. DUPLICATE ID ANALYSIS")
    duplicate_ids = df.groupBy("_Id").count().filter("count > 1")
    dup_count = duplicate_ids.count()
    print(f"\nNumber of duplicate IDs: {dup_count}")
    if dup_count > 0:
        print("\nSample duplicate IDs:")
        duplicate_ids.show(10)

    # 5. Class Distribution
    print_section("5. CLASS DISTRIBUTION")
    print("\nBadge Class distribution (1=Gold, 2=Silver, 3=Bronze):")
    class_dist = df.groupBy("_Class").count().orderBy("_Class")
    class_dist.show()

    # Check for invalid class values
    invalid_class = df.filter(~col("_Class").isin(1, 2, 3))
    invalid_class_count = invalid_class.count()
    print(f"\nRows with invalid class values (not 1, 2, or 3): {invalid_class_count:,}")
    if invalid_class_count > 0:
        print("\nSample invalid class values:")
        invalid_class.select("_Id", "_Class").show(10)

    # 6. TagBased Distribution
    print_section("6. TAG_BASED DISTRIBUTION")
    print("\nTagBased distribution:")
    df.groupBy("_TagBased").count().show()

    # Check for unexpected values
    print("\nDistinct TagBased values:")
    df.select("_TagBased").distinct().show()

    # 7. Date Range Analysis
    print_section("7. DATE RANGE ANALYSIS")
    date_range = df.select(
        min("_Date").alias("min_date"),
        max("_Date").alias("max_date")
    )
    print("\nDate range:")
    date_range.show(truncate=False)

    # Check for null dates
    null_dates = df.filter(col("_Date").isNull()).count()
    print(f"\nRows with null dates: {null_dates:,}")

    # 8. User ID Analysis
    print_section("8. USER ID ANALYSIS")
    print("\nUser ID statistics:")
    user_stats = df.select(
        min("_UserId").alias("min_user_id"),
        max("_UserId").alias("max_user_id")
    )
    user_stats.show()

    # Check for negative user IDs (could be system users)
    negative_users = df.filter(col("_UserId") < 0)
    neg_count = negative_users.count()
    print(f"\nRows with negative user IDs: {neg_count:,}")
    if neg_count > 0:
        print("\nSample negative user IDs:")
        negative_users.select("_Id", "_UserId", "_Name").show(10)

    # Check for null user IDs
    null_users = df.filter(col("_UserId").isNull()).count()
    print(f"\nRows with null user IDs: {null_users:,}")

    # 9. Badge Name Analysis
    print_section("9. BADGE NAME ANALYSIS")

    # Unique badge names
    unique_names = df.select("_Name").distinct().count()
    print(f"\nNumber of unique badge names: {unique_names:,}")

    # Top 20 most common badges
    print("\nTop 20 most common badges:")
    df.groupBy("_Name").count().orderBy(col("count").desc()).show(20)

    # Check for null or empty names
    null_names = df.filter(col("_Name").isNull()).count()
    empty_names = df.filter(trim(col("_Name")) == "").count()
    print(f"\nRows with null badge names: {null_names:,}")
    print(f"Rows with empty badge names: {empty_names:,}")

    # Check for names with leading/trailing whitespace
    whitespace_names = df.filter(
        col("_Name") != trim(col("_Name"))
    ).count()
    print(f"Rows with leading/trailing whitespace in name: {whitespace_names:,}")

    # 10. Data Type Summary
    print_section("10. DATA TYPE SUMMARY")
    print("\nExpected vs Actual types:")
    print("  _Id: Expected=Integer/Long, Actual=", df.schema["_Id"].dataType)
    print("  _UserId: Expected=Integer/Long, Actual=", df.schema["_UserId"].dataType)
    print("  _Name: Expected=String, Actual=", df.schema["_Name"].dataType)
    print("  _Date: Expected=Timestamp/String, Actual=", df.schema["_Date"].dataType)
    print("  _Class: Expected=Integer, Actual=", df.schema["_Class"].dataType)
    print("  _TagBased: Expected=Boolean/String, Actual=", df.schema["_TagBased"].dataType)

    # Summary
    print_section("PROFILING SUMMARY")
    print(f"""
    Total Records: {total_rows:,}
    Unique Badge Names: {unique_names:,}
    Duplicate IDs: {dup_count:,}
    Invalid Class Values: {invalid_class_count:,}
    Null Dates: {null_dates:,}
    Null User IDs: {null_users:,}
    Negative User IDs: {neg_count:,}
    Names with Whitespace Issues: {whitespace_names:,}
    """)


def main():
    """Main entry point."""
    print("=" * 60)
    print(" BADGES DATA PROFILING")
    print("=" * 60)

    print("\nInitializing Spark session...")
    spark = create_spark_session()

    # Load data
    file_path = "D:/Projects/Big-data-project/data/Badges.xml"
    print(f"\nLoading data from: {file_path}")

    df = load_badges_xml(spark, file_path)

    # Run profiling
    profile_badges(df)

    # Cleanup
    print("\n" + "=" * 60)
    print(" PROFILING COMPLETE")
    print("=" * 60)
    spark.stop()


if __name__ == "__main__":
    main()
