# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC # Demo: A Simple (but Quick) Query, Visualization, and Dashboard

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
Lesson Objectives</h2>
    <div class="instructions-div">
    <p>At the end of this lesson, you will be able to:</p>
    <ul>
    <li>Connect Databricks SQL queries to an existing Databricks SQL warehouse</li>  
    <li>Describe how to complete a basic query, visualization, and dashboard workflow using Databricks SQL</li>
  
    </ul></div>
    
    """, statements=None) 

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    <h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
The Landing Page</h2>
    <div class="instructions-div">
    <p>We are going to start by giving you an overview of the Databricks SQL landing page, the app switcher, and the sidebar menu. These features will help you navigate Databricks SQL as you progress through the course.</p>
    <p>The landing page is the main page for Databricks SQL. Note the following features of the landing page:</p>
    <ul>
    <li>Shortcuts</li>
    <li>Recents and Favorites</li>
    <li>Documentation Links</li>
    <li>Links to Release Notes</li>
    <li>Links to Blog Posts</li>
    </ul></div>""", statements=None) 

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
The Sidebar Menu and App Switcher</h2>
    <div class="instructions-div">
    <p>On the left side of the landing page is the sidebar menu and app switcher. </p>
        <ol>
            <li>Roll your mouse over the sidebar menu</li>
        </ol>
    <p>The menu expands. You can change this behavior using the "Menu Options" at the bottom of the menu.</p>
    <p>The app switcher allows you to change between Databricks SQL and the other apps available to you in Databricks. Your ability to access these apps is configured by your Databricks Administrator. Note that you can pin Databricks SQL as the default app when you login to Databricks.</p>
    </div>""", statements=None) 

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, add_username_input=True, instructions="""
<h2>
        <img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
        The Query Editor</h2>
    <div class="instructions-div">
        <p>There are a few ways to start a new query in Databricks SQL, but we are going to use the easiest.</p>
        <ol>
            <li>Click New --> Query in the sidebar menu</li>
        </ol>
        <p>This is the Query Editor in Databricks SQL. Note the following features of the Query Editor:</p>
        <ul>
            <li>Schema Browser</li>
            <li>Tabbed Interface</li>
            <li>Results View</li>
        </ul>
        <p>To run a query, we need to make sure we have a warehouse selected.</p>
        <ol start="2">
            <li>Click the drop-down in the upper-right corner to the left of the button labeled "Save"</li>
        </ol>
        <p>This drop-down lists the SQL Warehouses you have access to. It is possible you have access to only one Warehouse. If this is the case, select this Warehouse. If there are multiple Warehouses in the list, select the one that fits your organization's requirements.</p>
        
    <h3>Add Your Username</h3>
    <div class="instructions-div">
        <p>In order for the course curriculum to function correctly, we need to get your username.</p>
        <p>The text box below contains a query that will return your username.</p>
        <p>Complete the following:</p>
        <ol start="3">
            <li>Paste the code in the text box below into the Query Editor, and click Run in the Query Editor</li>
        </ol>
        </div>
    """, after_codebox_instructions="""
        <div class="instructions-div">
            <ol start="4">
                <li>Paste your username into the box below.</li>
                <li>Press Enter/Return on your keyboard</li>
            </ol>
            <p>We are going to use your username in future queries.</p>
        </div>
    """, statements="""SELECT current_user();""" )

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, add_catalog_input=True, instructions="""

    
        <h3>Create Your Catalog</h3>
        <p>You are now going to create a catalog that you will use throughout this course. In the text box below, input any catalog name you wish and press Enter/Return. This catalog will be used throughout the course.</p>
        """, statements=None )

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""
        <div class="instructions-div">
        <p>After you have input the catalog name, complete the following:</p>
            <ol>
                <li>Copy and paste the code </li>
                <li>Press Enter/Return on your keyboard</li>
            </ol>
            <p>We are going to use this schema name in future queries in <span class="monofont">USE</span> statements that will be generated for you in this course.</p>
        </div>
    """, statements=["CREATE CATALOG IF NOT EXISTS {catalog_name};", "USE CATALOG {catalog_name};", "CREATE SCHEMA IF NOT EXISTS dawd_v2;",] )

step.render()
step.execute() 

# COMMAND ----------

# %sql
# -- SOURCE_ONLY
# DROP TABLE IF EXISTS customers;
# CREATE TABLE customers AS
#   SELECT * FROM delta.`wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v03/retail-org/customers/customers_delta`;

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""

    <h2>
        <img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
        Lose the "Use"</h2>
    <div class="instructions-div">
        <p>Databricks SQL makes it easy to run queries without having to add USE statements. Just above the query input window, to the right of the Run button, there are two dropdowns.</p>
        <ol>
            <li>Drop open the first dropdown, and select your catalog name</li>
            <li>Drop open the second dropdown, and select the schema name "dawd_v2"</li>
        </ol>
        <p>Now, all queries will be run agains the DAWD_v2 schema in your catalog, without the need for USE statements.</p>
    </div>
    """, statements=None) 

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""

    <h2>
        <img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
        Now, It's Your Turn</h2>
    <div class="instructions-div">
        <p>In this part of the lesson, follow the instructions below to run a query, make a visualization from that query, and create a simple dashboard with that visualization.</p>
        <ol>
            <li>Ensure your catalog and the schema, "dawd_v2" are selected in the dropdowns above the query editor</li>
            <li>Paste the query below into the query editor, and click Run.</li>
        </ol>
    </div>
    """, after_codebox_instructions="""
    <div class="instructions-div">
        <p>You just created a table in the dawd_v2 schema called customers. We are going to discuss the <span class="monofont">CREATE</span> syntax later in the course.</p>
        <p>Note the following features of the results window:</p>
        <ul>
            <li>Number of Results Received</li>
            <li>Refreshed Time</li>
            <li>"+" button for adding visualizations, filters, and parameters</li>
        </ul>
        <p>When visualizations are added to your queries, they will also show up in the results window.</p>
    </div>
    """, statements=["DROP TABLE IF EXISTS customers;", """CREATE TABLE customers AS SELECT * FROM
    delta.`wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v03/retail-org/customers/customers_delta`;""", "SELECT * FROM customers;"]) 

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""

    <h2>
        <img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
        Add a Visualization</h2>
    <div class="instructions-div">
        <p>In the results window, we can see the data pulled from our schema. This is dummy data that shows customer information for a fake retail company.</p>
        <p>You are going to create a simple visualization, but please note that we are going to cover visualizations in much more detail later in the course. Complete the following steps:</p>
        <ol>
            <li>Hover over the "+" button in the results section of the query editor, and select Visualization</li>
            <li>Ensure "Bar" is selected under "Visualization Type"</li>
            <li>Drop down "X column", and select "state"</li>
            <li>Click "Add column" under "Y column", and select "*"</li>
            <li>Leave "Count" selected</li>
            <li>In the upper-left corner, click "Bar 1" and change the name to "Customer Count by State"</li>
            <li>Click "Save" in the lower-right corner</li>
        </ol>
        <p>The visualization is added to the query. Note that you did not have to perform any grouping or aggregation in the query itself, yet the visualization displays a count, grouped by state.</p>
        <p>Change the query so that it does not re-create the table each time it is run, and save the query:</p>
        <ol start="8">
            <li>Delete the <span class="monofont">DROP TABLE</span> and <span class="monofont">CREATE TABLE</span> commands in the query. You should be left with <span class="monofont">SELECT * FROM customers;</span></li>
            <li>Click the Save button above the SQL editor, and save the query as All Customers</li>
        </ol>
    </div>
    
    """, statements=None) 

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""

    <h2>
        <img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
        Create a Dashboard</h2>
    <div class="instructions-div">
        <p>We are going to create a dashboard that includes this visualization.</p>
    <ol>
        <li>Next to the name of the visualization (Customer Count by State), click the down-facing arrow.</li>
        <li>Click "Add to dashboard"</li>
        <li>Since there are no existing dashboards, click "Create new dashboard"</li>
        <li>Name the dashboard "Retail Customers", and click "Save"</li>
        <li>Click "Save and add"</li>
    </ol>
        <p>The dashboard has been created, and the visualization has been added to it. Let's take a look at the dashboard.</p>
    <ol start="6">
        <li>Click "Dashboards" in the sidebar menu</li>
        <li>Click the name of our dashboard, "Retail Customers"</li>
    </ol>   
    <p>Note the features of the dashboard:</p>
    <ul>
        <li>The ability to add Tags</li>
        <li>The "Share" button for sharing dashboards with others</li>
        <li>The "Schedule" button, where we can schedule the refresh of all visualizations in the dashboard</li>
        <li>The "Refresh" button for manually refreshing the dashboard</li>
    </ul>
    <p>We are going to go into a lot more detail later in the course, but note how easy it was to create a dashboard.</p>
    </div>
    
    """, statements=None) 

step.render()
step.execute() 

# COMMAND ----------

DA.cleanup(validate_datasets = False)
html = DA.publisher.publish(include_inputs=False, inputs_inline=True)
displayHTML(html)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
