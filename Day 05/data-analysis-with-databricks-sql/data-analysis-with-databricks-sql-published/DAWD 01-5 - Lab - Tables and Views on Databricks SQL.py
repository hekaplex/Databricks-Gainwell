# Databricks notebook source
# MAGIC %md 
# MAGIC # Lab: Tables and Views on Databricks SQL

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

    <h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
    Lab Objective</h2>
    <div class="instructions-div">
    <p>At the end of this lab, you will be able to:</p>
    <ul>
    <li>Use Databricks SQL to create tables and views</li>
    </ul></div>
    
    """) 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_validation(True, instructions="""

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Create External Tables</h2>
    <div class="instructions-div">
    <p>In this part of the lab, you are going to create an external table using a dataset located at 'wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v02/retail-org/sales/sales_delta'.</p>
    <p>Complete the following:</p>
    <ol>
        <li>Copy the query below into the query editor</li>
        <li>Make changes to the places marked "FILL_IN"</li>
        <li>Run the query in Databricks SQL</li>
        <li>Check your work by entering your answer to the question</li>
        <li>After pressing <span class="monofont">ENTER/RETURN</span>, green indicates a correct answer, and red indicates incorrect</li>
    </ol>
    </div>    
""", test_code = """USE {schema_name};
DROP TABLE IF EXISTS sales_external;
CREATE FILL_IN sales_external USING DELTA
    FILL_IN 'wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v02/retail-org/sales/sales_delta';
SELECT * FROM sales_external ORDER BY customer_id;
""", statements=["DROP TABLE IF EXISTS sales_external;",
                 "CREATE TABLE sales_external USING DELTA LOCATION 'wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v02/retail-org/sales/sales_delta';",
                 "SELECT * FROM sales_external ORDER BY customer_id;"], 
     label="""What is the value of <span class="monofont">customer_id</span> in the first row? """, 
     expected="12096776", 
     length=10)

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_validation(True, instructions="""

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Create Managed Table</h2>
    <div class="instructions-div">
    <p>The data for the table we created above is in a read-only object store. We want to copy the data into a local, managed table here in our workspace. In this part of the lab, we are going to create a managed table using a subquery, which we will learn about later in this course, to accomplish this.</p>
    <p>Complete the following:</p>
    <ol>
        <li>Copy the query below into the query editor</li>
        <li>Make changes to the places marked "FILL_IN"</li>
        <li>Run the query in Databricks SQL</li>
        <li>Check your work by entering your answer to the question</li>
        <li>After pressing <span class="monofont">ENTER/RETURN</span>, green indicates a correct answer, and red indicates incorrect</li>
    </ol>
    </div>""", test_code = """USE {schema_name};
FILL_IN TABLE sales USING DELTA AS
    SELECT * FROM sales_external;
DESCRIBE FILL_IN sales;
""", statements=["CREATE OR REPLACE TABLE sales AS SELECT * FROM sales_external;", "DESCRIBE EXTENDED sales;"], label="""What is the value of <span class="monofont">Owner</span>? """, expected="root", length=10)

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_validation(False, instructions="""

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Use Data Explorer</h2>
    <div class="instructions-div">
    <p>In this part of the lab, you will use the Data Explorer to find the answer to the question below.</p>
    <p>Complete the following:</p>
    <ol>
        <li>Click "Data" in the sidebar menu to go to the Data Explorer</li>
        <li>Check to the right of 'hive_metastore'. If your schema isn't selected, click on "default" and in the drop down menu, and select it.</li>
        <li>Select the table, "sales" from the list</li>
        <li>Feel free to explore the tabs to see more information about this table</li>
        <li>When you are ready, enter your answer to the question below</li>
        <li>After pressing <span class="monofont">ENTER/RETURN</span>, green indicates a correct answer, and red indicates incorrect</li>
    </ol>
    </div>


""", test_code = None, statements=None, label="""Under History, what was the operation associated with Version 1 of the table? """, expected="create or replace table as select", length=10)

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_validation(True, instructions="""

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Drop Table</h2>
    <div class="instructions-div">
    <p>We are going to keep the managed table, but we need to drop the external table.</p>
    <p>Complete the following:</p>
    <ol>
        <li>Copy the query below into the query editor</li>
        <li>Make changes to the places marked "FILL_IN" to drop the table, <span class="monofont">sales_external</span> from the schema</li>
        <li>Run the query in Databricks SQL</li>
        <li>The 'DESCRIBE' command should fail since metadata has been removed</li>
        <li>Check your work by entering your answer to the question</li>
        <li>After pressing <span class="monofont">ENTER/RETURN</span>, green indicates a correct answer, and red indicates incorrect</li>
    </ol>
    </div>""", test_code = """USE {schema_name};
FILL_IN TABLE sales_external;
DESCRIBE FILL_IN sales_external;
""", 
statements=["DROP TABLE sales_external;"], 
                                   
label="""Did the 'DESCRIBE' command fail? (yes or no) """, 
expected="yes", 
length=10)

# The DESCRIBE command is not checked in the code above because it is expected to fail.

step.render()
step.execute()

# COMMAND ----------

DA.cleanup()
html = DA.publisher.publish()
displayHTML(html)

