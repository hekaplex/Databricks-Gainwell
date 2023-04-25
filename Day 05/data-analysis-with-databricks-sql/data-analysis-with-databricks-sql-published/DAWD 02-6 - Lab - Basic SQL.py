# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC # Lab: Basic SQL

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

step = DA.publisher.add_step(False, instructions="""    

    <h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
    Lesson Objective</h2>
    <div class="instructions-div">
    <p>At the end of this lesson, you will be able to:</p>
    <ul>
    <li>Write basic SQL queries to subset tables using Databricks SQL</li>
    <li>Join multiple tables together to create a new table</li>
    <li>Aggregate data columns using SQL functions to answer defined business questions</li>
    </ul></div>
    
    """, statements=None) 

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_validation(True, instructions="""

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Retrieve Data</h2>
    <div class="instructions-div">
    <p>The statement we are using in this part of the lab implements <span class="monofont">SELECT</span>, <span class="monofont">SELECT ... AS</span>, <span class="monofont">GROUP BY</span>, and <span class="monofont">ORDER BY</span>. Note that <span class="monofont">FROM</span>, <span class="monofont">GROUP BY</span>, and <span class="monofont">ORDER BY</span> need to occur in a specific order, or an error will be thrown.</p>
    <p>Complete the following:</p>
    <ol>
        <li>Make the required changes to the query below</li>
        <li>Run the query in Databricks SQL</li>
        <li>Check your work by entering your answer to the question</li>
        <li>After pressing <span class="monofont">ENTER/RETURN</span>, green indicates a correct answer, and red indicates incorrect</li>
    </ol>
    </div>""", test_code="""USE {schema_name};
SELECT loyalty_segment, count(loyalty_segment) AS Count 
    FILL_IN customers 
    FILL_IN BY loyalty_segment 
    FILL_IN BY loyalty_segment;

""", statements="""SELECT loyalty_segment, count(loyalty_segment) AS Count 
    FROM customers 
    GROUP BY loyalty_segment 
    ORDER BY loyalty_segment;""", label="""How many customers are in loyalty_segment 0? """, expected="11097", length=10)

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_validation(True, instructions="""

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Use Column Expressions</h2>
    <div class="instructions-div">
    <p>The <span class="monofont">customers</span> table contains the column <span class="monofont">customer_name</span>, which has the same problem as the <span class="monofont">city</span> column had in the last lesson. The customer names are all in lower-case. Run a <span class="monofont">SELECT</span> query, using the <span class="monofont">initcap()</span> function to examine the results of using this function on the <span class="monofont">customer_name</span>.</p>
    <p>Complete the following:</p>
    <ol>
        <li>Make the required changes to the query below</li>
        <li>Run the query in Databricks SQL</li>
        <li>Check your work by entering your answer to the question</li>
        <li>After pressing <span class="monofont">ENTER/RETURN</span>, green indicates a correct answer, and red indicates incorrect</li>
    </ol>
    </div>""", test_code="""USE {schema_name};
SELECT FILL_IN(FILL_IN) AS Customer_Name 
    FROM customers
    ORDER BY customer_name DESC;

""", statements="""SELECT initcap(customer_name) AS Customer_Name 
    FROM customers
    ORDER BY customer_name DESC;""", label="""What is the last name of the last customer (alphabetically ) in the table? Ensure answer is in lower case """, expected="zyskowski", length=10)

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_validation(True, instructions="""

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Update Data</h2>
    <div class="instructions-div">
    <p>Let's go ahead and implement the changes we examined in the last section. Use an <span class="monofont">UPDATE</span> statement to change the data in the <span class="monofont">customers</span> table.</p>
    <p>Complete the following:</p>
    <ol>
        <li>Make the required changes to the query below</li>
        <li>Run the query in Databricks SQL</li>
        <li>Check your work by entering your answer to the question</li>
        <li>After pressing <span class="monofont">ENTER/RETURN</span>, green indicates a correct answer, and red indicates incorrect</li>
    </ol>
    </div>""", test_code="""USE {schema_name};
FILL_IN customers FILL_IN customer_name = initcap(customer_name);
SELECT * FROM customers;

""", statements=["UPDATE customers SET customer_name = initcap(customer_name);", "SELECT * FROM customers;"], label="""In which city is Bittner Engineering, Inc. located? """, expected="randolph", length=10)

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_validation(True, instructions="""

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Insert Data</h2>
    <div class="instructions-div">
    <p>We can implement a fifth loyalty segment in our <span class="monofont">loyalty_segments</span> table. We will use a <span class="monofont">unit_threshold</span> of 130 units.</p>
    <p>Complete the following:</p>
    <ol>
        <li>Make the required changes to the query below</li>
        <li>Run the query in Databricks SQL</li>
        <li>Check your work by entering your answer to the question</li>
        <li>After pressing <span class="monofont">ENTER/RETURN</span>, green indicates a correct answer, and red indicates incorrect</li>
    </ol>
    </div>""", test_code="""USE {schema_name};
FILL_IN INTO loyalty_segments
    (loyalty_segment_id, loyalty_segment_description, unit_threshold, valid_from, valid_to)
    FILL_IN
    (5, 'level_5', 130, current_date(), Null);
SELECT * FROM loyalty_segments;

""", statements=["""INSERT INTO loyalty_segments
    (loyalty_segment_id, loyalty_segment_description, unit_threshold, valid_from, valid_to)
    VALUES
    (5, 'level_5', 130, current_date(), Null);""", "SELECT * FROM loyalty_segments;"], label="""How many rows are in the table now? """, expected="5", length=10)

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_validation(True, instructions="""

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
CREATE OR REPLACE VIEW AS statement</h2>
    <div class="instructions-div">
    <p>In this part of the lab, we are going to create a view that contains only those customers in California (CA) who have a <span class="monofont">loyalty_segment</span> of 3.</p>
    <p>Complete the following:</p>
    <ol>
        <li>Make the required changes to the query below</li>
        <li>Run the query in Databricks SQL</li>
        <li>Check your work by entering your answer to the question</li>
        <li>After pressing <span class="monofont">ENTER/RETURN</span>, green indicates a correct answer, and red indicates incorrect</li>
    </ol>
    </div>""", test_code="""USE {schema_name};
CREATE OR REPLACE FILL_IN high_value_CA_customers AS
    SELECT * 
        FROM customers 
        WHERE state = 'FILL_IN'
        AND loyalty_segment = FILL_IN;
SELECT * FROM high_value_CA_customers;

""", statements=["""CREATE OR REPLACE TABLE high_value_CA_customers AS
    SELECT * 
        FROM customers 
        WHERE state = 'CA'
        AND loyalty_segment = 3;""", "SELECT * FROM high_value_CA_customers;"], label="""How many rows are in the view? """, expected="949", length=10)

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_validation(True, instructions="""

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Joins</h2>
    <div class="instructions-div">
    <p>Let's use a <span class="monofont">INNER JOIN</span>, <span class="monofont">GROUP BY</span>, <span class="monofont">ORDER BY</span>, and a function to calculate the total dollar amount of sales to various states in the sales table. Note that <span class="monofont">INNER JOIN</span> is the default join type, so we can just type <span class="monofont">JOIN</span>.</p>
    <p>Complete the following:</p>
    <ol>
        <li>Make the required changes to the query below</li>
        <li>Run the query in Databricks SQL</li>
        <li>Check your work by entering your answer to the question</li>
        <li>After pressing <span class="monofont">ENTER/RETURN</span>, green indicates a correct answer, and red indicates incorrect</li>
    </ol>
    </div""", test_code="""USE {schema_name};
FILL_IN customers.state, sum(total_price) AS Total FROM customers
    FILL_IN sales
    FILL_IN customers.customer_id = sales.customer_id
    GROUP BY customers.state
    ORDER BY Total DESC;


""", statements=["""CREATE OR REPLACE TABLE high_value_CA_customers AS
    SELECT * 
        FROM customers 
        WHERE state = 'CA'
        AND loyalty_segment = 3;""", "SELECT * FROM high_value_CA_customers;"], label="""Which state has the highest sales? """, expected="or", length=10)

step.render()
step.execute()

# COMMAND ----------

step = DA.publisher.add_validation(True, instructions="""

<h2><img class="image-icon-inline" src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/step-icon_small.png" alt="heading-icon" />
Use Aggregations</h2>
    <div class="instructions-div">
    <p>We have already used a few aggregate functions in the lab. Let's finish by running a simple aggregation to find the best price our suppliers have been able to provide on a specific piece of vintage electronics, a 5-disk CD changer. The best price means the lowest price, and we can use the <span class="monofont">min()</span> function to make this calculation.</p>
    <p>Complete the following:</p>
    <ol>
        <li>Make the required changes to the query below</li>
        <li>Run the query in Databricks SQL</li>
        <li>Check your work by entering your answer to the question</li>
        <li>After pressing <span class="monofont">ENTER/RETURN</span>, green indicates a correct answer, and red indicates incorrect</li>
    </ol>
    </div>""", test_code="""USE {schema_name};
SELECT FILL_IN(total_price)
    FROM sales;

""", statements=["""SELECT min(total_price) 
    FROM sales;"""], label="""What is the lowest price? (numbers only) """, expected="18", length=10)

step.render()
step.execute()

# COMMAND ----------

DA.cleanup()
html = DA.publisher.publish()
displayHTML(html)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
