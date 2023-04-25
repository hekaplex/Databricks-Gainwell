# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC # Demo: Integrations

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

    <h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
    Lesson Objective</h2>
    <div class="instructions-div">
    <p>At the end of this lesson, you will be able to:</p>
    <ul>
        <li>Describe how to connect Databricks SQL to ingestion tools like Fivetran.</li>
        <li>Describe how to connect Databricks SQL to visualization tools like Tableau, Power BI, and Looker.</li>
        <li>Identify the Databricks SQL REST API and Databricks CLI as tools for engineers to programmatically interact with Databricks SQL.</li>
    </ul></div>
    
    """, statements=None) 

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""

    <h2>
        <img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
        Connecting to Outside Sources of Data</h2>
    <div class="instructions-div">
        <p><span style="color:red;">Note: to connect to partner tools discussed in this part of the lesson, you will need to have privileges to create partner connections from your Databricks administrator.</span></p>
        <p></p>
        <ol start="1">
            <li>Click New --> Data in the sidebar menu</li>
        </ol>
        <p>Note the many connections that can be made on this page:</p>
        <ul>
            <li>Clicking "Upload data" will take you to the small file upload page discussed earlier in the course</li>
            <li>Clicking other buttons under "Native integrations" will open a Databricks notebook that can be configured to ingest data from the chosen source</li>
            <li>Clicking on any of the data sources under "Fivetran data sources" will open a wizard that will help you configure a connection to Fivetran. Note: You must have proper privileges to view the configuration wizard</li>
            <li>Clicking on "See all available ingest partners in Partner Connect" will open a new tab that will show additional partners that can easily be connected to Databricks SQL for data ingestion</li>
        </ul>
        <p>After looking at the ingestion partners available to you, close the tab to return to Databricks SQL.</p>
        </div>
    """, statements=None )

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""

    <h2>
        <img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
        Connecting to Partner BI and Visualization Tools</h2>
    <div class="instructions-div">
        <p><span style="color:red;">Note: to connect to partner tools discussed in this part of the lesson, you will need to have privileges to create partner connections from your Databricks administrator.</span></p>
        <p>You can configure connections to third-party BI and visualization tools using Partner Connect.</p>
        <ol>
            <li>Click Partner Connect near the bottom of the sidebar menu</li>
        </ol>
        <p>This page contains a huge number of connections that can be made to third-party tools, but let's concentrate on those for BI and visualization:</p>
        <ul>
            <li>Microsoft Power BI</li>
            <li>Tableau</li>
            <li>Looker (at the bottom of the page) -- note that this connection has to be made manually</li>
            <li>Others</li>
        </ul>
        <p>We won't actually make any of these connections in this course. However, the process is straightforward and well documented.</p>
        </div>
    """, statements=None )

step.render()
step.execute() 

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""

    <h2>
        <img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
        Personal Access Tokens (PATs)</h2>
    <div class="instructions-div">
        <p><span style="color:red;">Note: to connect to Databricks SQL using the CLI or API, you will need a personal access token (PAT). Your Databricks administrator needs to grant you the ability to create a PAT.</span></p>
        <p>The CLI and API are external methods for controlling Databricks SQL. In this portion of the lesson, I am going to show you where to create a PAT that must be used to work with the CLI or API. <span style="color:red;">This token must be kept secret. It is the same as a username and password and will grant its user the ability to act in your name on Databricks</span>. To create a PAT:</p>
        <ol>
            <li>Click your username in the upper-right corner of the page, and click "User Settings"</li>
            <li>Click "Personal access tokens"</li>
            <li>Click "+ Generate new token"</li>
            <li>Give the reason for the token in the comment section</li>
            <li>Change the lifetime of the token according to your needs (leaving this blank will set the token to never expire)</li>
            <li>Click Generate</li>
        </ol>
        <p>The token is created. This will be the only time you will have access to the token, so copy it now, and save it in a secure location. Once you click OK, the token cannot be viewed again.</p>
        <ol start="7">
            <li>Click the "copy" icon (it looks like two sheets of paper)</li>
            <li>Save the token to a secure location (again, you will not be able to view the token in the future within Databricks)</li>
            <li>Click OK to dismiss the dialog</li>
        </ol>
        <p>Using the API and CLI are beyond the scope of this class, but you can find more information about the Databricks SQL API <a href="https://docs.databricks.com/sql/api/index.html" target="_blank">here</a>. You can find more information about the Databricks CLI <a href="https://docs.databricks.com/dev-tools/databricks-sql-cli.html" target="_blank">here</a>.</p>
        </div>
    """, statements=None )

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
