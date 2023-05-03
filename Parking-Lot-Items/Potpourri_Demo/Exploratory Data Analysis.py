# Databricks notebook source
# MAGIC %md
# MAGIC #Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ##Spark

# COMMAND ----------

#Using Wine Data
white_wine = spark.read.csv("/FileStore/winequality_white.csv", sep=";", inferSchema=True, header=True)
red_wine = spark.read.csv("/FileStore/winequality_red.csv",sep=";", inferSchema=True, header=True)

# COMMAND ----------

display(white_wine)

# COMMAND ----------

display(red_wine)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Pandas

# COMMAND ----------

import pandas as pd
pd.read_csv("/dbfs/FileStore/winequality_white.csv",sep=';')

# COMMAND ----------

# MAGIC %pip install bamboolib

# COMMAND ----------

import bamboolib as bam

# COMMAND ----------

bam

# COMMAND ----------

#Do EDA and preprocessing from bamboolib, copy the code from it and use it across
import plotly.express as px
fig = px.histogram(df, x='pH')
fig

# COMMAND ----------

# MAGIC %md
# MAGIC ##SQL

# COMMAND ----------

# MAGIC %sql
# MAGIC Select * from feature_store_taxi_example.white_wine