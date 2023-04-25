# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC # Demo: Data Importing

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

    <h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
    Lesson Objective</h2>
    <div class="instructions-div">
    <p>At the end of this lesson, you will be able to:</p>
    <ul>
        <li>Identify small-file upload as a secure solution for importing small text files like lookup tables and quick data integrations.</li>
        <li>Use small-file upload in Databricks SQL to upload small text files securely as Delta tables.</li>
        <li>Add metadata to a table using Data Explorer.</li>
        <li>Import from object storage using Databricks SQL.</li>
    </ul></div>
    
    """, statements=None) 

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""

    <h2>
        <img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
        Uploading a Data File</h2>
    <div class="instructions-div">
        <p>In this lesson, we are going to use two methods of ingesting data into Databricks SQL. In this part of the lesson, we will upload a data file directly. There are many cases when it is beneficial to upload a data file directly to Databricks. When this is the case, Databricks provides Small File Upload, which gives you the opportunity to upload a subset of file types (currently only .csv) for quick analysis or for uploading simple tables, like lookup tables. In this part of the demo, I'm going to show you how to use this feature. To begin, we need a data file we can upload.</p>
        <ol start="1">
            <li>Click <a href="https://files.training.databricks.com/courses/data-analysis-with-databricks-sql/customers.csv">here</a> to download a .csv file</li>
            <li>Click New --> "File upload" in the sidebar menu</li>
        </ol>
        <p>In this simple interface, you can drop files directly, or you can browse for them on your local computer's filesystem. Note that you can only upload 10 files at a time and that the total file size cannot exceed 100 MB.</p>
        <ol start="3">
            <li>Drag and drop the customers.csv file (the one you downloaded in step 1) onto the page (or click "browse" and select the file)</li>
        </ol>
        <p>Note the features of the page that appears:</p>
        <ul>
            <li>The top area where we can select the location of the table.</li>
            <li>"Advanced attributes" where the delimiter, escape character, and other attributes can be selected</li>
            <li>The bottom area where a preview of the data is displayed, and we can make changes to columns.</li>
        </ul>
        <p>Complete the following:</p>
        <ol start="4">
            <li>Select your catalog name and the schema "dawd_v2"</li>
            <li>The table name defaults to the name of the file (minus the extension). Change the name of the table to "current_customers"</li>
            <li>We do not need the tax_id and tax_code columns. Click the down arrow next to these column names, and select "Exclude column"</li>
            <li>Click the column name, "postcode" and change the name to "postal_code"</li>
            <li>Change the data type for "valid_from" and "valid_to" to a timestamp by clicking the data type (to the left of the column name)</li>
            <li>Click "Create table" at the bottom of the page</li>
        </ol>
        <p>It may take a few seconds for the table to be created. We are then taken to the Data Explorer, and the new table is selected for us.</p>
    </div>
    """, statements=None )

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""

    <h2>
        <img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
        The Data Explorer</h2>
    <div class="instructions-div">
        <p>You can reach the Data Explorer at any time by clicking Data in the sidebar menu.</p>
        <p>Note the features of the Data Explorer:</p>
        <ul>
            <li>You can view any catalog, schema, and table that you have access to view</li>
        </ul>
        <ol>
            <li>Select the schema "dawd_v2"</li>
        </ol>
        <p>You can view information about this schema, including the type (Schema), the owner (your username), and the list of tables in the schema.</p>
        <ol start="2">
            <li>Click the table we just created, current_customers.</li>
        </ol>
        <p>There is additional information about the table.</p>
        <ul>
            <li>The data source format is Delta</li>
            <li>The current comment says, "Created by the file upload UI"</li>
            <li>The column names, data types, and comments for each column in the table</li>
            <li>Sample Data</li>
            <li>The Details tab shows additional information</li>
            <li>The Permissions tab shows currently granted privileges on the table</li>
            <li>Finally, the History tab shows actions that have been taken on the table. This is a feature that is available because this is a Delta table. We will talk more about this feature later in the course.</li>
        </ul>
        <p>Because you created the table, you are able to grant and revoke privileges on the table. While you can make changes to privileges using SQL, the Data Explorer allows you to grant and revoke various privileges using the UI.</p>
        <ol start="3">
            <li>Click Permissions</li>
        </ol>
        <p>Note that you have all privileges granted to you as the table's owner</p>
        <ol start="4">
            <li>Click Grant</li>
            <li>Select the checkbox next to ALL PRIVILEGES</li>
            <li>Click inside the box to add users and/or groups</li>
            <li>Select "account users"</li>
        </ol>
        <p>In order to allow for the "account users" group to SELECT and MODIFY this table, the group also needs to have USE CATALOG and USE SCHEMA granted on the catalog and schema, respectively. Let's do that now.
        <ol start="9">
            <li>Click Grant</li>
            <li>Click your catalog name</li>
            <li>Click Permissions</li>
            <li>Click Grant</li>
            <li>Select the checkbox next to USE CATALOG</li>
            <li>Click inside the box to add users and/or groups</li>
            <li>Select "account users"</li>
            <li>Click Grant</li>
            <li>Click the schema name "dawd_v2" and perform the same actions to grant USE SCHEMA to account users</li>
        </ol>
        <p>Let's revoke the privileges we granted to account users on our current_customers table.</p>
        <ol start="18">
            <li>Click the Tables tab, and click current_customers</li>
            <li>Click the Permissions tab</li>
            <li>Select the checkbox for account users</li>
            <li>Click Revoke</li>
            <li>Click Revoke in the warning that appears</li>
        </ol>
    </div>
    """, statements=None )

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""

    <h2>
        <img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
        Quick Dashboard</h2>
    <div class="instructions-div">
        <p>The Data Explorer makes it really easy to create a quick dashboard, which you can use as-is, or make changes, as needed.</p>
        <p>Complete the following:</p>
        <ol>
            <li>With the table, current_customers, selected in the Data Explorer, hover over the Create menu, and select Quick dashboard</li>
        </ol>
        <p>The Create quick dashboard dialog appears, where we can choose columnns to include in the dashboard. Quick dashboards have specific visualizations that are created based on the data type of the column chosen.</p>
        <ol start="2">
            <li>Select the state, units_purchased, loyalty segment, and valid_from columns</li>
            <li>Click Create</li>
        </ol>
        <p>A dashboard is created with visualizations that fit the data type of each column. These visualizations and the dashboard itself can be customized, and we will talk about that later in the course.</p>
    </div>
    """, statements=None )

step.render()
step.execute() 

# COMMAND ----------

# step = DA.publisher.add_step(False, instructions="""    

# <h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
# Creating a Table with Data from External Location</h2>
#     <div class="instructions-div">
#     <p><span style="color:red;">Note: the instructions we are completing in this part of the lesson require that you have the "<span class='monofont'>SELECT</span>" and "<span class='monofont'>MODIFY</span>" privileges on "<span class='monofont'>ANY FILE</span>". If you do not have these privileges, you will not be able to run the code in this section.</span></p>
#     <p>There is a dataset, called Sales, that is currently in an object store. Running the command below creates an external table that is associated with this dataset. "External" means the data remains in the object store location, and the table that is created keeps a reference to the location in its metadata. Any changes to the table will affect the data in the object store.</p>
#     <ol>
#         <li>Run the code below:</li>
#     </ol>
#     </div>
#     """, after_codebox_instructions="""
#     <div class="instructions-div">
#         <p>We can query the data just like any other table in the schema. The keyword in the <span class="monofont">CREATE</span> command is <span class="monofont">LOCATION</span>. This specifies the location of the data, but it does not copy that data anywhere. Also, if the table is dropped, the data will remain in the location specified.</p>
#     </div>
    
#     """, statements=["DROP TABLE IF EXISTS external_table;", """CREATE TABLE external_table
#     LOCATION 'wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v03/retail-org/sales/sales_delta';""",
#     "SELECT * FROM external_table;"]) 

# step.render()
# step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Creating Tables with Data from an Object Store</h2>
    <div class="instructions-div">
    <p><span style="color:red;">Note: the instructions we are completing in this part of the lesson require that you have the "<span class='monofont'>SELECT</span>" and "<span class='monofont'>MODIFY</span>" privileges on "<span class='monofont'>ANY FILE</span>". If you do not have these privileges, you will not be able to run the code in this section.</span></p>
    <p>With Unity Catalog, you will, most likely, want to create managed tables, meaning the data will be copied from an external object store into the metastore location created by your Databricks administrator.</p>
    <ol>
        <li>Go to the Query Editor by clicking SQL Editor in the sidebar menu</li>
        <li>Ensure your catalog and the "dawd_v2" schema are selected to the right of the Run button</li>
        <li>Run the code below:</li>
    </ol>
    </div>
    """, after_codebox_instructions="""
    <div class="instructions-div">
        <p>We are creating an empty table without a <span class="monofont">LOCATION</span> specified. The table's data will be copied and stored in the metastore location (specified by an administrator when the metastore was created). The code also uses <span class="monofont">COPY INTO</span>. This command is useful for copying data from an object store into a table. We are going to talk a lot more about this command later in the course.</p>
    <p>Running DESCRIBE EXTENDED shows us metadata about the table we created. Note the following:</p>
    <ul>
        <li>The catalog name</li>
        <li>The database (schema) name</li>
        <li>The table type: MANAGED</li>
        <li>The Location, which is the location of the metastore</li>
        <li>The Provider, delta, meaning this is a Delta type table</li>
    </ul>
    <p>Generally, the external object store will not be available to the public like it is in this example. Granting access to external locations is not covered in this course.</p>
    </div>
    
    """, statements=["DROP TABLE IF EXISTS managed_table;", "CREATE TABLE managed_table;","""COPY INTO managed_table
    FROM 'wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v03/retail-org/loyalty_segments/loyalty_segment.csv'
    FILEFORMAT = csv
    FORMAT_OPTIONS ('mergeSchema' = 'true',
                  'delimiter' = ',',
                  'header' = 'true')
    COPY_OPTIONS ('mergeSchema' = 'true');""", "DESCRIBE EXTENDED managed_table;"]) 

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
