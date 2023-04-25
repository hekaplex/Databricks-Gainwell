# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC # Demo/Lab: Lakehouse SQL

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

    <h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
    Lesson Objective</h2>
    <div class="instructions-div">
    <p>At the end of this lesson, you will be able to:</p>
    <ul>
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

# step = DA.publisher.add_validation(False, instructions="""

# <h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
# Grant/Revoke Privileges Using SQL</h2>
#     <div class="instructions-div">
#     <p>In this portion of the lab, we are going to grant all privileges on our brand new table, <span class="monofont">sales_stream</span>, to all users in the workspace. We are then going to immediately revoke MODIFY from all users. By default, Databricks SQL has two groups: "admins" and "users", but Databricks admins can add more groups, as needed.</p>
#     <p>Complete the following:</p>
#     <ol>
#         <li>Run the query below in Databricks SQL</li>
#         <li>Enter your answer to the question</li>
#         <li>After pressing <span class="monofont">ENTER/RETURN</span>, green indicates a correct answer, and red indicates incorrect</li>
#     </ol>
#     </div>""", test_code = """USE {schema_name};
# GRANT SELECT ON TABLE `sales_stream` TO `users`;
# SHOW GRANT ON `sales_stream`;


# """, statements=["GRANT SELECT ON TABLE `sales_stream` TO `users`;", 
#                  "SHOW GRANT ON `sales_stream`;"], 
#    label="""How many privileges does the group "users" have on <span class="monofont">sales_stream</span>? """, 
#    expected="1", length=10)

# step.render()
# # We cannot execute this statement via notebooks
# # step.execute()

# COMMAND ----------

# step = DA.publisher.add_validation(False, instructions="""

# <h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
# Grant/Revoke Privileges Using the Data Explorer</h2>
#     <div class="instructions-div">
#     <p>We are now going to use the Data Explorer to perform the same tasks as above. Follow the instructions to find the answer to the question below.</p>
#     <p>Complete the following:</p>
#     <ol>
#         <li>Click "Data" in the sidebar menu to go to the Data Explorer</li>
#         <li>Click "Default" to open the drop down, and select your schema</li>
#         <li>Select the table, "<span class="monofont">sales_stream</span>" from the list</li>
#         <li>Select the "permissions" tab</li>
#         <li>Click "Grant"</li>
#         <li>Click inside the input box and select "All Users"</li>
#         <li>Click inside the input box again to dismiss the list</li>
#         <li>Select the checkbox next to "<span class="monofont">MODIFY</span>" in the list of permissions</li>
#         <li>Click "OK"</li>
#     </ol>
#     <p>Note which privileges have now been granted on the table <span class="monofont">sales_stream</span>. To revoke these privileges, complete the following:</p>
#     <ol start="10">
#         <li>Select the checkboxes next to all privileges granted to "users"</li>
#         <li>Click "Revoke"</li>
#         <li>Read the warning, and click "Revoke"</li>
#         <li>When you are ready, enter your answer to the question below</li>
#         <li>After pressing <span class="monofont">ENTER/RETURN</span>, green indicates a correct answer, and red indicates incorrect</li>
#     </ol>
    
#     </div>


# """, test_code=None, statements=None, 
#      label="""How many privileges have you been granted to you on your schema? """, 
#      expected="6", length=10)

# step.render()
# step.execute()

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
