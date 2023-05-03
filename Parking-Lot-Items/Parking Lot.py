# Databricks notebook source
# MAGIC %md
# MAGIC ### Join Types

# COMMAND ----------

Emp = [(1,"Ram",-2,"2019","11","M",2000), 	
    (2,"Pooja",2,"2011","30","F",5000), 	
    (3,"Shyam",1,"2011","20","M",2000), 	
    (4,"Madhavi",2,"2006","20","F",3000), 
    (5,"Brown",2,"2011","30","",-2), 
      (6,"Brown",2,"2011","40","",-2) 
  ]	
Emp_Columns = ["emp_id","name","superior_emp_id","year_joined", 
       "emp_dept_id","gender","salary"]	

Emp_Dataframe = spark.createDataFrame(data = Emp, schema = Emp_Columns)	

Emp_Dataframe.printSchema()	
display(Emp_Dataframe)	

Dept = [("Marketing",10), 
    ("Finance",20), 
    ("IT",30),
    ("Sales",40)
  ]
Dept_Columns = ["dept_name","dept_id"]	
Dept_Dataframe = spark.createDataFrame(data = Dept, schema = Dept_Columns)	
Dept_Dataframe.printSchema()	
display(Dept_Dataframe)
  


# COMMAND ----------

# Using outer join 
display(Emp_Dataframe.join(Dept_Dataframe, Emp_Dataframe.emp_dept_id ==  Dept_Dataframe.dept_id,"outer"))



# COMMAND ----------

# Using inner join 	

display(Emp_Dataframe.join(Dept_Dataframe, Emp_Dataframe.emp_dept_id ==  Dept_Dataframe.dept_id,"inner"))	

# COMMAND ----------

# Using full join 	

display(Emp_Dataframe.join(Dept_Dataframe, Emp_Dataframe.emp_dept_id ==  Dept_Dataframe.dept_id,"full")	)	


# COMMAND ----------

# Using full outer join 	

display(Emp_Dataframe.join(Dept_Dataframe, Emp_Dataframe.emp_dept_id ==  Dept_Dataframe.dept_id,"fullouter"))

# COMMAND ----------

# Using left join 	

display(Emp_Dataframe.join(Dept_Dataframe, Emp_Dataframe.emp_dept_id ==  Dept_Dataframe.dept_id,"left"))	


# COMMAND ----------

# Using left outer join 	

display(Emp_Dataframe.join(Dept_Dataframe, Emp_Dataframe.emp_dept_id ==  Dept_Dataframe.dept_id,"leftouter"))

# COMMAND ----------

# Using right join 	

display(Emp_Dataframe.join(Dept_Dataframe, Emp_Dataframe.emp_dept_id ==  Dept_Dataframe.dept_id,"right"))	

# COMMAND ----------

# Using right outer join	

display(Emp_Dataframe.join(Dept_Dataframe, Emp_Dataframe.emp_dept_id ==  Dept_Dataframe.dept_id,"rightouter"))	


# COMMAND ----------

# Using left semi join	

display(Emp_Dataframe.join(Dept_Dataframe, Emp_Dataframe.emp_dept_id ==  Dept_Dataframe.dept_id,"leftsemi"))	


# COMMAND ----------

# Using left anti join  	

display(Emp_Dataframe.join(Dept_Dataframe, Emp_Dataframe.emp_dept_id ==  Dept_Dataframe.dept_id,"leftanti"))	

# COMMAND ----------

# Using self join	
from pyspark.sql.functions import col
display((
    Emp_Dataframe
    .alias("emp1")
    .join(Emp_Dataframe.alias("emp2")
    ,col("emp1.superior_emp_id") == col("emp2.emp_id"),"inner")
    .select(col("emp1.emp_id"),col("emp1.name"),col("emp2.emp_id")
    .alias("superior_emp_id"),col("emp2.name")
    .alias("superior_emp_name"))
    ))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split

# COMMAND ----------

# MAGIC %sql
# MAGIC --third paramter is limit on number of splits
# MAGIC --[] allows for multiple split criteria
# MAGIC SELECT split('oneAtwoBthreeC', '[ABC]')
# MAGIC UNION ALL
# MAGIC SELECT split('oneAtwoBthreeC', '[ABC]', -1)
# MAGIC UNION ALL
# MAGIC SELECT split('oneAtwoBthreeC', '[ABC]', 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ToDDL in Python

# COMMAND ----------

from pyspark.sql import *

# create a sample dataframe
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
df = spark.createDataFrame(data, ["name", "age"])

def hack_toDDL_Python(df):
    schema_ddl = ""
    for col in df.schema:
        schema_ddl += f"{col.name} {col.dataType.typeName()} " + \
                    f"{'' if col.nullable else 'NOT NULL'}" + ", "
    schema_ddl = schema_ddl.strip().rstrip(",")

    # print the schema DDL
    return(schema_ddl)


# COMMAND ----------

hack_toDDL_Python(Emp_Dataframe)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Partitions
# MAGIC Coalesce partitions: When you have too many partitions, it can lead to slow processing times due to increased overhead. 
# MAGIC 1. You can coalesce the RDD into fewer partitions to reduce overhead and improve performance. 
# MAGIC 1. Repartition RDD: If you need to increase the number of partitions for better parallelism, you can repartition the RDD. 
# MAGIC 1. Set partitioner: If you want to use a custom partitioner for your RDD, you can set it using the partitionBy() method. 

# COMMAND ----------

# create an RDD with 10 partitions
rdd = sc.parallelize(range(100), 10)

# coalesce the RDD into 5 partitions
coalesced_rdd = rdd.coalesce(5)


# COMMAND ----------

# create an RDD with 10 partitions
rdd = sc.parallelize(range(100), 10)

# repartition the RDD into 20 partitions
repartitioned_rdd = rdd.repartition(20)


# COMMAND ----------

# create a key-value RDD
rdd = sc.parallelize([(0, "a"), (1, "b"), (2, "c"), (3, "d"), (4, "e")])
partitioned_rdd = rdd.partitionBy(2, lambda x: x % 2)


# COMMAND ----------


