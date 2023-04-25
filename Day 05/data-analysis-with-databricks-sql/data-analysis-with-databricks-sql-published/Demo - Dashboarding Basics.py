# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC # Demo: Dashboarding Basics

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

    <h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
    Lesson Objectives</h2>
    <div class="instructions-div">
    <p>At the end of this lesson, you will be able to:</p>
    <ul>
        <li>Create a dashboard using multiple existing visualizations from Databricks SQL Queries.</li>
        <li>Create and use queries and dashboards with filters and parameters to allow stakeholders to customize results and visualizations.</li>
        <li>Share queries and dashboards with stakeholders with appropriate permissions.</li>
        <li>Organize and label workspace objects, queries and dashboards within the Databricks SQL platform.</li>
    </ul></div>
    
    """, statements=None) 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Create Table and Load Data for this Lesson</h2>
    <div class="instructions-div">
    <p>Complete the following:</p>
    <ol>
        <li>Ensure your catalog and the schema, "dawd_v2" are selected in the dropdowns above the query editor</li>
        <li>Run the query below</li>
    </ol>
    </div>
    
    """, statements=["DROP TABLE IF EXISTS sales;", """CREATE TABLE sales AS SELECT * FROM delta.`wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v03/retail-org/sales/sales_delta`;"""]) 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Dashboards</h2>
    <div class="instructions-div">
    <p>We are now going to combine all the visualizations we created above into a dashboard that will display them all at once and that we can put on a refresh schedule to keep the data that underlies each visualization up-to-date. In a future lesson, we will talk about setting a refresh schedule and subscribing stakeholders to the dashboard's output, so they can always have the newest information.</p>
    <p>Complete the following:</p>
    <ol>      
        <li>Click "Dashboards" in the sidebar menu</li>
        <li>Click "Create Dashboard"</li>
        <li>Name the dashboard "Retail Organization"</li>
        <li>Click "Add", and click "Visualization"</li>
    </ol>
    <p>We are presented with a list of queries that have been saved. Although we named our queries based on the type of visualization we made, it makes more sense to name a query based on what it does.</p>
    <ol start="5">
        <li>Click "Count Total Sales"</li>
        <li>Leave "Select existing visualization" selected, and drop down the list directly below. Note we have the results table from our query and our counter visualization, "Total Sales" available to us. Select "Total Sales"</li>
        <li>Change "Title" to "Total Sales"</li>
        <li>Optional: write a description</li>
        <li>Click "Add to dashboard"</li>
        <li>Repeat steps 4-9 with the "Sales Over Three Months" query ("Sales by Month" and "Sales by Product Category"), "Count Customers by State" query ("Most Active States"), and "All Customers" query ("Customer Locations")</li>
    </ol>
    <p>You should have five visualizations in the dashboard</p>
    <ol start="11">
        <li>Click "Add", then "Text Box", and type "# Customers and Sales" in the "Edit:" box</li>
    </ol>
    <p>Note that text boxes support Markdown.</p>
    <ol start="12">
        <li>Click "Add to dashboard"</li>
        <li>Optional: Move the visualizations around by clicking and dragging each one</li>
        <li>Optional: Resize each visualization by dragging the lower-right corner of the visualization</li>
        <li>Optional: Click "Colors" to change the color palette used by visualizations in the dashboard</li>
        <li>Click "Done Editing" in the upper-right corner</li>
        <li>Run every query and refresh all visualizations all at once by clicking "Refresh"</li>
    </ol> 
    </div>
    
    """, statements=None) 

step.render() 
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Parameterized Queries</h2>
    <div class="instructions-div">
    <p>Before we leave this lesson, let's talk about a customization feature we can apply to our queries to give them more flexibility. Query parameters allow us to make changes to our queries without requiring new code to be written.</p>
    <p>Complete the following:</p>
    <ol>
        <li>Go back to the Query Editor and start a new query</li>
        <li>Paste the query below into the editor and save the query as "Get Product Category"</li>
</ol>
    </div>
    
    """, statements="SELECT DISTINCT product_category FROM sales;") 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h3>Writing a Query with Parameters</h3>
    <div class="instructions-div">
    <p>Now that we have a query that pulls all product categories from the <span class="monofont">Sales</span> table, let's use this query as a parameter in a second query.</p>
    <p>Complete the following:</p>
    <ol start="3">
        <li>Start a new query, and paste the code below in the editor</li>
    </ol>
    <p>Note that the query has empty single quotes.</p>
    <ol start="4">
        <li>Place your cursor in-between the single quotes, hover over the "+" symbol, and click "Parameter"</li>
        <li>Input "category" for the "Keyword" field</li>
        <li>Drop down "Type" and choose "Query Based Dropdown List"</li>
        <li>For "Query" choose the query we created above: "Get Product Category"</li>
        <li>Click "Add Parameter"</li>
        <li>Save the query as "Total Sales by Product Category"</li>
    </ol>
    <p>Note two things: First, we now have a set of double curly braces that contain the word "category". This is where are query parameter was inserted. Finally, note the dropdown list we how have just above the query results window.</p>
    <ol start="10">
        <li>Open the dropdown list and choose a Category from the list</li>
        <li>Click "Apply Changes"</li>
    </ol>
    <p>The query is rerun with the chosen product category replacing the location of the query parameter in the query. Thus, we see the Total Sales of the Category we chose.</p>
    </div>
    
    """, statements="""SELECT sum(total_price) AS Total_Sales FROM sales
    WHERE product_category = '';""") 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Organizing Queries, Alerts, and Dashboards</h2>
    <div class="instructions-div">
    <p>Queries, alerts (we talk about alerts in a few minutes), and dashboards can be organized in a folder structure to help you keep track of where your work is located.</p>
    <ol start="1">
        <li>Click on "Workspace" in the sidebar menu</li>
    </ol>
    <p>You are taken to your home folder. You can see the dashboard we created in this lesson and any other workspace objects that you have saved.</p>
    <ol start="2">
        <li>Click "Create"</li>
    </ol>
    <p>Note all of the items you can create from this menu.</p>
    <ol start="3">
        <li>Select "Folder"</li>
        <li>Name the folder "My Great Project", and click "Create"</li>
        <li>After the folder is created, drag and drop the dashboard you just created in this lesson onto the new folder</li>
        <li>Click the folder name, and verify that the dashboard has moved to this folder</li>
    </ol>
    <p>You can nest folders however you want, whether by project, queries that act on bronze data, silver data, or however else you see fit.</p> 
    """, statements=None)


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
