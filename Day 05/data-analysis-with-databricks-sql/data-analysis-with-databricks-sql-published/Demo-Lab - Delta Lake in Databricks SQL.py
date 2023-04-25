# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC # Demo: Delta Lake in Databricks SQL

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

    <h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
    Lesson Objectives</h2>
    <div class="instructions-div">
    <p>At the end of this lesson, you will be able to:</p>
    <ul>
        <li>Describe Delta Lake as a tool for managing data files.</li>
        <li>Describe the benefits of Delta Lake within the Lakehouse.</li>
        <li>Identify common differences between Delta Lake and popular enterprise data warehouses.</li>
        <li>Summarize best practices for creating and managing schemas (databases), tables, and views on Databricks.</li>
        <li>Use Databricks to create, use, and drop schemas (databases), tables, and views.</li>
        <li>Optimize commonly used Delta tables for storage using built-in techniques.</li>
        <li>Identify ANSI SQL as the default standard for SQL in Databricks.</li>
        <li>Identify Delta- and Unity Catalog-specific SQL commands as the preferred data management and governance methods of the Lakehouse.</li>
        <li>Identify common differences between common source environment SQL and Databricks Lakehouse SQL.</li>
        <li>Describe the basics of the Databricks SQL execution environment.</li>
        <li>Apply Spark- and Databricks-specific built-in functions to scale intermediate and advanced data manipulations.</li>
        <li>Review query history and performance to identify and improve slow-running queries.</li>
        <li>Access and clean silver-level data.</li>
        <li>Describe the purpose of SQL user-defined functions (UDFs).</li>
        <li>Create and apply SQL functions within the context of the medallion architecture.</li>
        <li>Use Databricks SQL to ingest data</li>

    </ul></div>
    
    """, statements=None) 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Delta Lake</h2>
    <div class="instructions-div">
    <p>Delta Lake is the optimized storage layer that provides the foundation for storing data and tables in the Databricks Lakehouse Platform. Delta Lake is open source software that extends Parquet data files with a file-based transaction log for ACID transactions and scalable metadata handling. Delta Lake allows you to easily use a single copy of data for both batch and streaming operations, providing incremental processing at scale.</p>
    
    <p>Here are the benefits of using Delta Lake in the Databricks Lakehouse:</p>
    <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/courses/data-analysis-with-databricks-sql/Delta+Lake+High+Level+Overview_crop.png" style="width:100%">
    <p>Here are a handful of differences between using Delta Lake in the Lakehouse and other data warehouse solutions:</p>
    <ul>
        <li>ACID transactions for data stored in data lakes</li>
        <li>A single source of truth</li>
        <li>Time travel</li>
        <li>Schema evolution/enforcement</li>
        <li>Open architecture</li>
    </ul>
    </div>
    
    """, statements=None) 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Best Practices</h2>
    <div class="instructions-div">
    <p>Databricks recommends using catalogs to provide segregation across your organization’s information architecture. Often this means that catalogs can correspond to software development environment scope, team, or business unit.</p>
    <p>In the graphic below, there is a catalog for each environment of the software development lifecycle for each business unit. In this way data is organized and easily promoted from one stage of development to the next.</p>
    <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/courses/data-analysis-with-databricks-sql/uc-catalogs-scaled.png" style="width:100%">
    <p></p>
    </div>
    
    """, statements=None) 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Create Tables and Load Data for this Lesson</h2>
    <div class="instructions-div">
    <p>Complete the following:</p>
    <ol>
        <li>Ensure your catalog and the schema, "dawd_v2" are selected in the dropdowns above the query editor</li>
        <li>Run the query below</li>
    </ol>
    </div>
    
    """, statements=["DROP TABLE IF EXISTS customers;", """CREATE TABLE customers AS 
    SELECT * FROM delta.`wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v03/retail-org/customers/customers_delta`;""", "SELECT * FROM customers;","DROP TABLE IF EXISTS source_suppliers;", """CREATE TABLE source_suppliers AS 
    SELECT * FROM delta.`wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v03/retail-org/suppliers/suppliers_delta`;""", "SELECT * FROM source_suppliers;","DROP TABLE IF EXISTS suppliers;", """CREATE TABLE suppliers AS 
    SELECT * FROM delta.`wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v03/retail-org/suppliers/suppliers_delta`;""","DELETE FROM suppliers;"]) 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Create Schema with One Table</h2>
    <div class="instructions-div">
    <span style="color:red;">Note: to create a schema (database) as discussed in this part of the lesson, you will need to have privileges to create schemas from your Databricks administrator.</span>
    <p>I want to show you how to create a new schema (database), and we will add one table to the schema.</p>
    <ol>
        <li>Click SQL Editor in the sidebar menu to go to the SQL editor</li>
        <li>Run the code below.</li>
    </ol>
    </div>""",after_codebox_instructions="""
        <div class="instructions-div">
    <p>The code runs five statements. The first drops the schema just in case we are running this more than once. The second creates the schema. The schema will be created in the catalog chosen in the dropdown to the right of the Run button.</p>
    <p>In the third statement, we override the chosen schema in the dropdown above the query editor and <span class='monofont'>USE</span> the schema we just created.</p>
    <p>In the last three statements, we create a simple table, fill it with one row of data, and <span class='monofont'>SELECT</span> from it.</p>

    </div>
    
    """, statements=["DROP SCHEMA IF EXISTS temporary_schema CASCADE;",
                     "CREATE SCHEMA IF NOT EXISTS temporary_schema;", 
                     "CREATE OR REPLACE TABLE temporary_schema.simple_table (width INT, length INT, height INT);",
                     "INSERT INTO temporary_schema.simple_table VALUES (3, 2, 1);",
                     "SELECT * FROM temporary_schema.simple_table;"]) 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h3>View the Table's Metadata</h3>
    <div class="instructions-div">
    <ol>
        <li>Run the code below to see information about the table we just created.</li>
    </ol>
    <p>Note that the table is a managed table, and the underlying data is stored in the metastore's default location. When we drop this table, our data will be deleted. Note also that this is a Delta table.</p>
    </div>
    
    """, statements="DESCRIBE EXTENDED temporary_schema.simple_table;") 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h3>DROP the Table</h3>
    <div class="instructions-div">
    <ol start="5">
        <li>Run the code below to drop the table.</li>
    </ol>
    </div>
    
    """, statements="DROP TABLE IF EXISTS temporary_schema.simple_table;") 

step.render()
step.execute()

# COMMAND ----------

# step = DA.publisher.add_step(False, instructions="""    

# <h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
# External Tables</h2>
#     <div class="instructions-div">
#     <p>There is a dataset, called Sales, that is currently in an object store. Running the command below creates an external table that is associated with this dataset.</p>
#     <ol start="5">
#         <li>Run the code below.</li>
#     </ol>
#     <p>The table's data is stored in the external location, but the table itself is registered in the metastore. We can query the data just like any other table in the schema.</p>
#     </div>
    
#     """, statements=["""CREATE TABLE external_table
#     LOCATION 'wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v02/retail-org/sales/sales_delta';""",
#     "SELECT * FROM external_table;"]) 

# step.render()
# step.execute()

# COMMAND ----------

# step = DA.publisher.add_step(False, instructions="""    

# <h3>View Table Information</h3>
#     <div class="instructions-div">
#     <p>Let's look at the table's information</p>
#     <ol start="6">
#         <li>Run the code below.</li>
#     </ol>
#     <p>Note that the table's type is <span class="monofont">EXTERNAL</span>, and the table's location points to the object store. Dropping this table will not affect the data in this location.</p>
#     </div>
    
#     """, statements="DESCRIBE EXTENDED external_table;") 

# step.render()
# step.execute()

# COMMAND ----------

# step = DA.publisher.add_step(False, instructions="""    

# <h3>Dropping External Tables</h3>
#     <div class="instructions-div">
#     <p>The command below will drop the table from the schema.</p>
#     <ol start="7">
#         <li>Run the code below to drop the table.</li>
#     </ol>
#     <p>Note that we dropped the table, so we won't be able to query the data using the kind of <span class="monofont">SELECT</span> query you may be used to using.</p>
#     </div>
    
#     """, statements="DROP TABLE IF EXISTS external_table;") 

# step.render()
# step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h3>Dropping External Tables Does Not Delete Data</h3>
    <div class="instructions-div">
    <ol start="8">
        <li>Run the code below.</li>
    </ol>
    <p>Even though we dropped the table, we are still able to query the data directly because it still exists in the object store.</p>
    <p>Now, this is one of the coolest features of Databricks SQL. We've talked about how the use of schemas and tables is just an organizational contruct. The data files located in this location can be queried directly, even though they are not part of a table or schema. We use tables and schemas simply to organize data in a way familiar to you.</p>
    </div>
    
    """, statements="SELECT * FROM delta.`wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v02/retail-org/sales/sales_delta`;")

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Views</h2>
    <div class="instructions-div">
    <p>Views can be created from other views, tables, or data files. In the code below, we are creating a view from data in a data file in an external object store.</p>
    <ol>
        <li>Run the code below.</li>
    </ol>
    </div>""", after_codebox_instructions="""<div class="instructions-div">
    <p>The view gives us all the sales that totaled more than 10,000. If new rows are added to the Sales data file, the view will update every time it's run.</p>
    </div>
    
    """, statements=[
    """CREATE OR REPLACE VIEW high_sales AS
    SELECT * FROM delta.`wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v02/retail-org/sales/sales_delta` 
        WHERE total_price > 10000;""",
    "SELECT * FROM high_sales;"]) 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    
<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
    Time Travel</h2>
   <h3>SELECT on Delta Tables</h3>
    <div class="instructions-div">
    <p>In the next few queries, we are going to look at commands that are specific to using <span class="monofont">SELECT</span> on Delta tables.</p>
    <p>Delta tables keep a log of changes that we can view by running the command below.</p>
    <ol start="1">
        <li>Run the code below.</li>
    </ol>
    </div>""", after_codebox_instructions="""
    <div class="instructions-div">
    <p>After running <span class="monofont">DESCRIBE HISTORY</span>, we can see that we are on version number 0 and we can see a timestamp of when this change was made.</p>
    </div>
    
    """, statements="DESCRIBE HISTORY customers;") 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

   <h3>SELECT on Delta Tables -- Updating the Table</h3>
    <div class="instructions-div">
    <p>We are going to make two seperate changes to the table.</p>
    <ol start="2">
        <li>Run the code below.</li>
    </ol>
    </div>""", after_codebox_instructions="""
    <div class="instructions-div">
    <p>The code uses two <span class="monofont">UPDATE</span> statements to make two changes to the table. We also reran our <span class="monofont">DESCRIBE HISTORY</span> command, and note that the updates are noted in the log with new timestamps. All changes to a Delta table are logged in this way.</p>
    </div>
    
    """, statements=["UPDATE customers SET loyalty_segment = 10 WHERE loyalty_segment = 0;","UPDATE customers SET loyalty_segment = 0 WHERE loyalty_segment = 10;","DESCRIBE HISTORY customers;"]) 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

   <h3>SELECT on Delta Tables -- VERSION AS OF</h3>
    <div class="instructions-div">
    <p>We can now use a special predicate for use with Delta tables: <span class="monofont">VERSION AS OF</span></p>
    <ol start="3">
        <li>Run the code below.</li>
    </ol>
    <p>By using <span class="monofont">VERSION AS OF</span>, we can <span class="monofont">SELECT</span> from specific versions of the table. This feature of Delta tables is called "Time Travel," and it is very powerful.</p>
    <p>We can also use <span class="monofont">TIMESTAMP AS OF</span> to <span class="monofont">SELECT</span> based on a table's state on a specific date and time, and you can find more information on this in the documentation.</p>
    </div>""", after_codebox_instructions="""<div class="instructions-div">
        <p>Our select on the table returns the results from version 2 of the table, which had changed loyalty segments that were equal to 0 to 10. We see this reflected in the results set.</p>
        </div>
    """, statements="SELECT loyalty_segment FROM customers VERSION AS OF 1;") 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    
<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
    Restore a Table to a Past State</h2>
    <div class="instructions-div">
    <p>If we wish to restore a table to a previous version or timestamp, we can use the <span class="monofont">RESTORE</span> command.</p>
    <ol>
        <li>Run the code below.</li>
    </ol>
    </div>""", after_codebox_instructions="""
    <div class="instructions-div">
    <p>After running the code, we can see the restore added to the log, and if we ran a <span class="monofont">SELECT</span> on the table, it would contain the state that it had in version 1.</p>
    </div>
    
    """, statements=["RESTORE TABLE customers TO VERSION AS OF 1;","DESCRIBE HISTORY customers;"]) 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_validation(False, instructions="""

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Ingest Data Using COPY INTO</h2>
    <div class="instructions-div">
    <p>COPY INTO is often used to ingest streaming data. Files that have previously been ingested are ignored, and new files are ingested every time COPY INTO is run. We can set this up by configuring COPY INTO to point to a directory and not specifying and patterns or files.</p>
    <p>In the query below, we are going to a create an empty Delta table, so we will specify the schema. We will then use COPY INTO to fill the table with data. After making changes to the query, run it more than once, and note that the number of rows does not change. As new .json files are added to the directory, they will be ingested into the table. Note: for this example, there are no new data files being placed in the directory, so the number of rows will not change.</p>
    <p>Make changes to the query below so that COPY INTO pulls data from the directory 'wasb://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v02/retail-org/sales_stream/sales_stream_json' and configure the file format as 'JSON'.</p>
    <p>Complete the following:</p>
    <ol>
        <li>Make the required changes to the query below</li>
        <li>Run the query in Databricks SQL</li>
        <li>Check your work by entering your answer to the question</li>
        <li>After pressing <span class="monofont">ENTER/RETURN</span>, green indicates a correct answer, and red indicates incorrect</li>
    </ol>
    """, test_code = """CREATE OR REPLACE TABLE sales_stream (5minutes STRING, 
                                      clicked_items ARRAY&lt;ARRAY&lt;STRING&gt;&gt;, 
                                      customer_id STRING, 
                                      customer_name STRING, 
                                      datetime STRING, 
                                      hour BIGINT, 
                                      minute BIGINT, 
                                      number_of_line_items STRING, 
                                      order_datetime STRING, 
                                      order_number BIGINT, 
                                      ordered_products ARRAY&lt;ARRAY&lt;STRING&gt;&gt;, 
                                      sales_person DOUBLE, 
                                      ship_to_address STRING
);
COPY INTO sales_stream 
    FROM 'FILL_IN'
    FILEFORMAT = FILL_IN;
SELECT * FROM sales_stream ORDER BY customer_id;

""", statements=["""CREATE OR REPLACE TABLE sales_stream (5minutes STRING, 
                          clicked_items ARRAY<ARRAY<STRING>>, 
                          customer_id STRING, 
                          customer_name STRING, 
                          datetime STRING, 
                          hour BIGINT, 
                          minute BIGINT, 
                          number_of_line_items STRING, 
                          order_datetime STRING, 
                          order_number BIGINT, 
                          ordered_products ARRAY<ARRAY<STRING>>, 
                          sales_person DOUBLE, 
                          ship_to_address STRING
);""", """COPY INTO sales_stream 
    FROM 'wasb://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v02/retail-org/sales_stream/sales_stream_json'
    FILEFORMAT = JSON;""", "SELECT * FROM sales_stream ORDER BY customer_id;"], label="""What is the value of <span class="monofont">customer_id</span> in the first row? """, expected="10060379", length=10)

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_validation(False, instructions="""

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Ingest Data Using MERGE INTO</h2>
    <div class="instructions-div">
    <p>Certainly, there are times when we want to insert new data but ensure we don't re-insert matched data. This is where we use <span class="monofont">MERGE INTO</span>. <span class="monofont">MERGE INTO</span> will merge two tables together, but you specify in which column to look for matched data and what to do when a match is found. Let's run the code and examine the command in more detail.</p>
    <ol start="5">
        <li>Run the code below.</li>
    </ol>
    <p>We are merging the <span class="monofont">source_suppliers</span> table into the <span class="monofont">suppliers</span> table. After <span class="monofont">ON</span> keyword, we provide the columns on which we want to look for matches. We then state what the command should do when a match is found. In this example, we are inserting the row when the two columns are not matched. Thus, if the columns match, the row is ignored. Notice that the count is the exact same as the original table. This is because the two tables have the exact same data, since we overwrote the <span class="monofont">suppliers</span> table with the <span class="monofont">source_suppliers</span> table earlier in the lesson.</p>
    </div>
    """, test_code = """MERGE INTO suppliers
    USING source_suppliers
    ON suppliers.SUPPLIER_ID = source_suppliers.SUPPLIER_ID
    WHEN NOT MATCHED THEN INSERT *; 
    SELECT count(*) FROM suppliers;""", 
         statements=["""MERGE INTO suppliers
    USING source_suppliers
    ON suppliers.SUPPLIER_ID = source_suppliers.SUPPLIER_ID
    WHEN NOT MATCHED THEN INSERT *;"""], label="""What is the value of <span class="monofont">num_affected_rows</span>? """, expected="15", length=10)

step.render()
step.execute()

# COMMAND ----------

DA.cleanup(validate_datasets = False)
html = DA.publisher.publish()
displayHTML(html)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
