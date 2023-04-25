# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC # Demo: Data Visualizations

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

    <h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
    Lesson Objective</h2>
    <div class="instructions-div">
   
    <p>At the end of this lesson, you will be able to:</p>
    <ul>
        <li>Identify Databricks SQL as a smart, in-platform visualization tool for the Lakehouse.</li>
        <li>Create basic, schema-specific visualizations using Databricks SQL.</li>
        <li>Compute and display summary statistics using data visualizations.</li>
        <li>Interpret summary statistics within data visualizations for business reporting purposes.</li>
        <li>Describe the further customized visualizations that can be created by Databricks SQL.</li>
        <li>Explore the different types of visualizations that can be created using Databricks SQL.</li>
        <li>Create customized data visualizations to aid in data storytelling.</li>
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
    
    """, statements=["DROP TABLE IF EXISTS sales;", """CREATE TABLE sales AS SELECT * FROM delta.`wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v03/retail-org/sales/sales_delta`;""", "DROP TABLE IF EXISTS customers;", """CREATE TABLE customers AS 
    SELECT * FROM delta.`wasbs://courseware@dbacademy.blob.core.windows.net/data-analysis-with-databricks/v03/retail-org/customers/customers_delta`;""", "SELECT * FROM customers;"]) 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
The Counter</h2>
    <div class="instructions-div">
    <p>The Counter visualization is one of the simplest visualizations in Databricks SQL. It displays a single number by default, but it can also be configured to display a "goal" number. In this example, we are going to configure a sum of completed sales, along with a "Sales Goal." The query calculates a sum of total sales and also provides a hard-coded sales goal column.</p>
    <p>Complete the following:</p>
    <ol>
        <li>Open a new query tab, and run the query below</li>
        <li>Save the query as "Count Total Sales"</li>
    </ol>
    <p>Visualizations are stored with the queries that generate data for them. Although we probably could pick a better name than "Counter", this will help us when we build our dashboard later in this lesson. Note also that we can have multiple visualizations attached to a single query</p>
    <ol start="3">
        <li>In the query results section, hover over the "+" symbol, and click "Visualization"</li>
        <li>Select "Counter" as the visualization type</li>
        <li>For "Counter Label" type "Total Sales"</li>
        <li>For "Counter Value Column" make sure the column "Total_Sales" is selected</li>
        <li>For "Target Value Column" choose "Sales Goal"</li>
    </ol>
    <p>Note that we can configure the counter to count rows for us if we did not aggregate our data in the query itself.</p>
    <ol start="8">
        <li>Click the "Format" tab</li>
        <li>Optional: Change the decimal character and thousands separator</li>
        <li>"Total Sales" is a dollar figure, so add "$" to "Formatting String Prefix"</li>
        <li>Turn the switch, "Format Target Value" to on</li>
        <li>Click "Save" in the lower-right corner</li>
        <li>Click the name of the visualization (the name of the tab) and change the name to "Total Sales"</li>
        <li>Make sure the query is Saved</li>
    </ol>
    <p></p>
    </div>
    
    """, statements="""SELECT sum(total_price) AS Total_Sales, 3000000 AS Sales_Goal 
    FROM sales;""") 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
The Bar Chart</h2>
    <div class="instructions-div">
    <p>One of the most often used visualizations in data analytics is the Bar Chart. Databricks SQL supports an variety of customization options to make bar charts look beautiful. In this example, we are going to configure a bar chart </p>
    <p>Complete the following:</p>
    <ol>
        <li>Open a new query tab, and run the query below</li>
        <li>Save the query as "Sales Over Three Months"</li>
        <li>In the query results section, hover over the "+" symbol, and click "Visualization"</li>
        <li>Select "Bar" as the visualization type</li>
        <li>For "X Column" choose "Month"</li>
        <li>For "Y Columns" click "Add column" and select "Total Sales" and "Sum"</li>
        <li>Click "Add column" again and select "Total Sales" and "Count"</li>
        <li>Click the "Y Axis" tab and type "Dollars" in the "Name" field (Left Y Axis)</li>
        <li>Click the "Series" tab and type "Total Sales" in the first "Label" field</li>
        <li>Type "Number of Sales" in the second "Label" field and change "Type" to "Line"</li>
        <li>Click "Save" in the lower-right corner</li>
        <li>Click the name of the visualization (the name of the tab) and change the name to "Sales by Month"</li>
        <li>Make sure the query is Saved</li>
    </ol>
    <p>As we can see from the visualization, the number of sales in August and October was low, but the dollar amounts of those sales was high. The opposite is true in September.</p>
    </div>
    
    """, statements="""SELECT customer_name, total_price AS Total_Sales, date_format(order_date, "MM") AS Month, product_category 
    FROM sales 
    WHERE order_date >= to_date('2019-08-01') 
    AND order_date <= to_date('2019-10-31');""") 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
The Stacked Bar Chart</h2>
    <div class="instructions-div">
    <p>We can glean more data from the same query by adding a second visualization.</p>
    <p>Complete the following:</p>
    <ol>
        <li>Hover over the "+" symbol, and click "Visualization"</li>
        <li>Change "Visualization Type" to "Bar"</li>
        <li>For "X Column" choose "product_category"</li>
        <li>Add two Y Columns and change both to "Total_Sales". Change the first to "Average" and the second to "Min"</li>
        <li>Change "Stacking" to "Stack"</li>
        <li>On the "X Axis" tab, change the name to "Product Category"</li>
        <li>On the "Y Axis" tab, change the name to "Dollars"</li>
        <li>On the "Series" tab, change the first row Label to "Average Sales" and the second row to "Minimum Sales"</li>
        <li>Click "Save" in the lower-right corner</li>
        <li>Click the name of the visualization (the name of the tab) and change the name to "Sales by Product Category"</li>
        <li>Make sure the query is Saved</li>
    </ol>
    <p>This visualization shows that, although the "Reagate" category has the highest minimum sales figure, it has the lowest average.</p>
    </div>
    
    """, statements=None) 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Maps - Choropleth</h2>
    <div class="instructions-div">
    <p>Databricks SQL has two map visualizations you can use to plot address and geolocation data: choropleth and markers. The choropleth map visualization uses color to show the count of a criterion within a specific geographic area. In this example, we are going to use customer address data to plot the number of customers in each U.S. state.</p> 
    <p>To make a choropleth map, complete the following:</p>
    <ol>
        <li>Open a new query tab, and run the query below</li>
        <li>Save the query as "Count Customers by State"</li>
        <li>Hover over the "+" symbol, and click "Visualization"</li>
        <li>Select "Map (Choropleth)" as the visualization type</li>
        <li>In the "General" tab, change "Map" to "USA", "Key Column" to state, "Target Field" to "USPS Abbreviation", and "Value Column" to "count(customer_id)"</li>
        <li>Click "Save" in the lower-right corner</li>
        <li>Click the name of the visualization (the name of the tab) and change the name to "Most Active States"</li>
        <li>Make sure the query is Saved</li>
    </ol>
    
    </div>
    
    """, statements="""SELECT state, count(customer_id) FROM customers
    GROUP BY state;""") 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Maps - Markers</h2>
    <div class="instructions-div">
    <p>The Map (Markers) visualization type plots points on a map that signify a specific location. In this example, we have latitude and longitude data for our customer locations. We will use this to plot those locations on a map.</p> 
    <p>Complete the following:</p>
    <ol>
        <li>Open a new query tab, and run the query below</li>
        <li>Save the query as "All Customers"</li>
        <li>Hover over the "+" symbol, and click "Visualization"</li>
        <li>Select "Map (Markers)" as the "Visualization Type"</li>
        <li>In the General tab, change "Latitude Column" to "lat", "Longitude Column" to "lon", and "Group By" to "state"</li>
        <li>On the "Format" tab, enable tooltips and type "&lcub;&lcub;customer_name&rcub;&rcub;" in the "Tooltip template" field</li>
    </ol>
    <p>Note: Because we are on a 2x-Small Warehouse, do not uncheck "Cluster Markers" in the "Styles" tab. The map refresh process will take a very long time to update.</p> 
    <ol>
        <li>Click "Save" in the lower-right corner</li>
        <li>Click the name of the visualization (the name of the tab) and change the name to "Customer Locations"</li>
        <li>Make sure the query is Saved</li>
    </ol> 
    
    </div>
    
    """, statements="""SELECT * FROM customers;""") 

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
