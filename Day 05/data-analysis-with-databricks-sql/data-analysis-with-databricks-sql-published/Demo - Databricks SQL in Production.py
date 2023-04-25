# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC # Demo: Databricks SQL in Production

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

    <h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
    Lesson Objectives</h2>
    <div class="instructions-div">
    <p>At the end of this lesson, you will be able to:</p>
    <ul>
        <li>Compare and contrast SQL automation capabilities between Databricks SQL and the rest of the Databricks Lakehouse Platform.</li>
        <li>Describe how refresh schedules affect dashboards.</li>
        <li>Build dashboards with automatic refresh schedules.</li>
        <li>Implement query alerts in common scenarios.</li>
        <li>Develop production-grade analytics solutions to create flat gold-level tables.</li>
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
    
    """, statements=["DROP TABLE IF EXISTS gym_logs;", """CREATE TABLE gym_logs AS SELECT * FROM delta.`wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v03/gym_logs/gym_logs_small_delta`;""", "SELECT * FROM gym_logs;","DROP TABLE IF EXISTS sales;", """CREATE TABLE sales AS SELECT * FROM delta.`wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v03/retail-org/sales/sales_delta`;"""]) 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
SQL Automation</h2>
    <div class="instructions-div">
    <p>Using Databricks SQL, you can automate many tasks that make working in the Lakehouse much easier. The automations that are available to you in Databricks SQL are:</p>
    <ul>
        <li>Query refresh schedules</li>
        <li>Dashboard refresh schedules</li>
        <li>Alerts</li>
    </ul>
    <p>These automations are configured within Databricks SQL and are independent of any automations in the rest of the Lakehouse, meaning, they can affect data anywhere in the Lakehouse, but they do not use Workflows or Jobs. Furthermore, they use SQL Warehouses, as opposed to clusters.</p>
    </div>""", statements=None) 

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Query Refresh Schedule</h2>
    <div class="instructions-div">
    <p>You can use scheduled query executions to keep your dashboards updated or to enable routine alerts. Let's make a query and put it on a refresh schedule.</p>
    <ol>
        <li>Run the query below.</li>
        <li>Name the query by clicking "New Query" and typing “Gym Logs”</li>
        <li>Click "Save"</li>
    </ol>
    <p>The query needs to be saved with a descriptive name, so we can reference it later in this lesson. To refresh this query automatically:</p>
    <ol start="4">
        <li>Click "Never" next to "Refresh Schedule" (bottom-right corner of query window)</li>
        <li>Change the dropdown to something other than "Never"</li>
        <li>Change "End" to tomorrow's date</li>
    </ol>
    <p><span style="color:red">WARNING: If the refresh rate is less than the SQL Warehouse 'Auto Stop' parameter, the Warehouse will run indefinitely.</span></p>
    </div>""", statements="""SELECT gym, count(*) number_of_visits 
    

FROM gym_logs
GROUP BY gym
ORDER BY gym;
""") 

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Alerts</h2>
    <div class="instructions-div">
    <p>Alerts allow you to configure notifications when a field returned by a scheduled query meets a specific threshold. Although we just configured a refresh schedule for our query, the Alert runs on its own schedule.</p>
    <p>To create an Alert:</p>
    <ol start="7">
        <li>Click "Alerts" in the sidebar menu</li>
        <li>Click "Create Alert"</li>
        <li>From the Query dropdown, select our query: "Gym Logs"</li>
        <li>Use the dropdown to change the "Value" column to <span class="monofont">number_of_visits</span> and change "Threshold" to 4</li>
        <li>Change "Refresh" to Never. We are going to trigger this alert manually</li>
    </ol>   
    <p>The default destination is the user’s email address. The alert is triggered when the count of the top row in the query’s results is greater than 1.</p>
    <p>Let's add some data to trigger the alert.</p>
    <ol start="12">
        <li>Run the code below.</li>
    </ol>
    <p>This code will ingest the remaining gym log data from the object store. This will increase the number of gym visits past our threshold and trigger the alert.</p>
    <p>Something to note with regard to configuring Alerts and Refresh Schedules: Every time they run, the SQL Warehouse will start (if it's stopped), run the query, and go into an idle state. Once the Auto Stop time has expired, the SQL Warehouse will stop. If the refresh schedule is set to a lower time limit than the SQL Warehouse's Auto Stop time, the Warehouse will never stop. This may increase costs.</p>
    <ol start="13">
        <li>Delete the alert and change the refresh schedule back to "Never".</li>
    </ol>
    </div>
    
    """, statements="""COPY INTO gym_logs 
    FROM 'wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v03/gym_logs/gym_logs_json/20191202_2.json'
    FILEFORMAT = JSON;""") 

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Sharing Queries</h2>
    <div class="instructions-div">
    <p>We can share queries with other members of the team:</p>
    <ol start="13">
        <li>Back in the Query Editor, click "Share"</li>
    </ol>   
    <p>The "Manage Permissions" dialogue appears. If you do not have permission to change settings, all options will be greyed out. Note that, as the owner of the query, you have "Can manage" permissions. You can share the query with users and groups who are configured in your workspace. These users and groups can have either "Can run" or "Can edit" permissions. Those with "Can edit" permissions can also run the query. In order to allow "Can edit" permissions, the Credentials drop down must be changed to "Run as Viewer". Click inside the input box, and a dropdown will show all users and groups with whom the query can be shared.</p>
    <ol start="14">
        <li>Select a user or group</li>
        <li>Select either "Can run" or "Can edit" permissions</li>
        <li>Close the dialogue</li>
    </ol>   
    <p>Note that any "Can edit" permissions that were granted must be revoked before the credential type for the query can be changed back to "Run as owner".</p>
    </div>
    
    """, statements=None) 

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Sharing Dashboards</h2>
    <div class="instructions-div">
    <p>Sharing dashboards is exactly the same as sharing queries. Click "Share" from any dashboard to update sharing permissions.</p>
       
    </div>
    
    """, statements=None) 

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Refreshing Dashboards and Sharing Results</h2>
    <div class="instructions-div">
    <p>We can set a refresh schedule for a dashboard and, optionally, share the results with others.</p>
    <ol start="17">
        <li>From any dashboard, click "Schedule"</li>
        <li>Drop down "Refresh" and select a refresh interval</li>
        <li>Optionally, select a SQL Warehouse to use to refresh the dashboard</li>
        <li>Set any Subscribers to be notified of dashboard results</li>
        <li>Ensure that "Enabled" is set to on</li>
        <li>Click "Save"</li>
    </ol>    
    <p>When you are finished with the dashboard refresh schedule, go ahead and disable it.</p>
    <p><span style="color:red">WARNING: If the Dashboard refresh interval is less than the SQL Warehouse 'Auto Stop' parameter, the Warehouse will run indefinitely.</span></p>
    </div>
    
    """, statements=None) 

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Creating Flat, Gold Level Tables</h2>
    <div class="instructions-div">
    <p>In almost all cases, data does not arrive in the Lakehouse clean and ready for use. We have to make changes to the data. In this portion of the lesson, we are going to make a couple of simple changes to the <span class="monofont">sales</span> table, but we don't want to make the changes in place. We will make a <span class="monofont"> sales_gold </span> table, so our raw ingested data remains unchanged.</p>
    <ol start="1">
        <li>Run the query below to get all data from the <span class="monofont">sales</span> table</li>
    </ol>        
    """, statements="SELECT * FROM sales;")


step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

   <h3>Use CTAS to Create sales_gold Table</h3>
    <div class="instructions-div">
    <p>If you look at the results, you can see that the <span class='monofont'>product</span> column is a <span class='monofont'>struct</span>. We want to flatten this into multiple columns to make the data easier to use. Also, note that the <span class='monofont'>customer_name</span> column is sometimes upper-case and sometimes lower-case. Let's make this more consistent.</p>
    <ol start="2">
        <li>Run the code below.</li>
    </ol>
    </div>""", after_codebox_instructions="""<div class="instructions-div">
        <p>We used dot notation to expand the <span class='monofont'>struct</span> into it's parts (example, <span class='monofont'>product.name</span>). We then ran <span class='monofont'>UPDATE</span> to change the <span class='monofont'>customer_name</span> to all lower-case, and then capitalized the first letter in each word.</p>
        </div>""", statements=["""CREATE OR REPLACE TABLE sales_gold AS
	SELECT customer_id, customer_name, order_date, product.curr, product.id, product.name, product.price, product.qty, product.unit
	FROM sales;""","UPDATE sales_gold SET customer_name = initcap(lower(customer_name));","SELECT * FROM sales_gold;"])

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
