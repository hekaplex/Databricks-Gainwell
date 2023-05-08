## Parking Lot Items Completed

### Shown in Lecture/Demo
1. ~~streaming queryname in SQL~~
1. ~~JDBC/ODBC Examples~~
1. ~~show all native spark file formats~~
1. ~~Shallow Copy~~
1. ~~Passing SQL Query as a variable~~
    1. ~~Avoid reruns of same code~~
    1. ~~spark.sql vs Spark SQL~~
    1. ~~resuable data pipeline~~
### External Documentation    
1. [DBR Run time link](https://docs.databricks.com/release-notes/runtime/releases.html)
1. [External library integration](https://docs.databricks.com/libraries/index.html)
1. ![Shift Left Thinking](https://www.stickyminds.com/sites/default/files/shared/2018-12-10%20ArthurHicken%20The%20Shift-Left%20Approach%20to%20Software%20Testing%20image3.png)
    1. [Databricks CI/CD](https://docs.databricks.com/dev-tools/ci-cd/ci-cd-jenkins.html)
1. [VIOLATION of constraints options](https://docs.databricks.com/delta-live-tables/sql-ref.html)
### New Examples in Repo
1. [Partitions - Spark Configuration](https://github.com/hekaplex/Databricks-Gainwell/blob/master/Parking-Lot-Items/Parking%20Lot.py)
1. [Markdown cheat sheet](https://github.com/hekaplex/Databricks-Gainswell/Parking-Lot-Items/Markdown-Examples.MD )
1. [Venn diagrams of semi, anti join](https://github.com/hekaplex/Databricks-Gainwell/blob/master/Parking-Lot-Items/Parking%20Lot.py)
1. [third parameter on split](https://github.com/hekaplex/Databricks-Gainwell/blob/master/Parking-Lot-Items/Parking%20Lot.py)
1. [create DDL schema from object schema](https://github.com/hekaplex/Databricks-Gainwell/blob/master/Parking-Lot-Items/Parking%20Lot.py)
1. [UDAF sample in Python, SQL](https://github.com/hekaplex/Databricks-Gainwell/blob/master/Parking-Lot-Items/Parking%20Lot.py)

### Demos/Examples Requested - READY
1. ~~Spark Pandas UDF example~~
1. ~~where is my regstered SQL UDF?~~
1. MLFlow
1. Feature Store
1. Schema Evolution
    1. [AutoLoader](https://docs.databricks.com/ingestion/auto-loader/schema.html)
    1. [Spark](https://docs.databricks.com/external-data/json.html)
1. Dolly Demo
1. ~~VS Code Python programmatic Spark session build~~
    1. ~~CI/CD between three environments~~
    1. ~~no dbx - YAML & CI/CD~~
    1. ~~UnityCatalog API is current alternate but need SparkSession~~
1. [Mini-Dolly Instructions for Training](https://github.com/databrickslabs/dolly#getting-started-with-response-generation)
1. non-PAT based authentication: ETA?  [OAuth user-to-machine (U2M) authentication](https://docs.databricks.com/dev-tools/auth.html#id10)
## Parking Lot TODO
1. Speed test on ASP 4.1
    1. Photon vs non-Photon
    1. speed test count distinct vs approx_count_distinct
1. Fix on Adv. DE Notebook 3.2
1. Fix on Adv. DE Notebook 5.1
1. Fix on Adv. DE Notebook 6.2
1. fix external events_extended