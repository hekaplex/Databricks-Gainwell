# Databricks notebook source
# MAGIC %md 
# MAGIC # Lab: Dashboards

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

    <h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
    Lab Objective</h2>
    <div class="instructions-div">
    <p>At the end of this lab, you will be able to:</p>
    <ul>
        <li>Create an additional dashboard using multiple visualizations from Databricks SQL Queries.</li>
        <li>Parameterize a dashboard to customize visualizations.</li>

    </ul></div>
    
    """) 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_validation(False, instructions="""

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon">
Create a Dashboard</h2>
    <div class="instructions-div">    
    <p>Let's combine the visualizations we just made into a dashboard.</p>
    <p>Complete the following:</p>
    <ol>      
        <li>Click "Dashboards" in the sidebar menu</li>
        <li>Click the blue "Create Dashboard" button</li>
        <li>Name the dashboard "Company Information" and click the "Save" button</li>
        <li>Click the blue "Add" button, and click Visualization"</li>
        <li>In the dropdown, select "Boxplot"</li>
        <li>Click the drop down under "Select existing visualization" and select "Price by Product" (which you created earlier)</li>
        <li>Change "Title" to "Prices"</li>
        <li>Click the blue "Add to dashboard" button, and the Boxplot appears in your dashboard</li>
        <li>Repeat steps 4-8 with the "Funnel" query ("Customer Funnel")</li>
        <li>Click "Add", then "Text Box", and type "# Miscellaneous Information" in the "Edit:" field</li>        
        <li>Click "Add to dashboard"</li>
        <li>Optional: Move the visualizations around by clicking and dragging each one</li>
        <li>Optional: Resize each visualization by dragging the lower-right corner of the visualization</li>
        <li>Optional: Click "Colors" to change the color palette used by visualizations in the dashboard</li>
        <li>Click "Done Editing" in the upper-right corner</li>
        <li>Run every query and refresh all visualizations all at once by clicking "Refresh"</li>
        <li>Check your work by entering your answer to the question below</li>
        <li>Hint: Hover over each Product Category to get the statistics in the Boxplot Chart</li>
        <li>After pressing <span class="monofont">ENTER/RETURN</span>, green indicates a correct answer, and red indicates incorrect</li>
    </ol> 
    </div>


""", test_code=None, statements=None, label="""Which product category has the lowest minimum price? """, expected="rony", length=10)

step.render()
step.execute()

# COMMAND ----------

DA.cleanup(validate_datasets = False)
html = DA.publisher.publish()
displayHTML(html)

