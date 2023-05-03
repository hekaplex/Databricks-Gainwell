# Databricks notebook source
# MAGIC %md # MLflow quickstart: inference
# MAGIC
# MAGIC This notebook shows how to load a model previously logged to MLflow and use it to make predictions on data in different formats. The notebook includes two examples of applying the model:
# MAGIC * as a scikit-learn model to a pandas DataFrame
# MAGIC * as a PySpark UDF to a Spark DataFrame
# MAGIC   
# MAGIC ## Requirements
# MAGIC * This notebook requires Databricks Runtime 6.4 or above, or Databricks Runtime 6.4 ML or above. You can also use a Python 3 cluster running Databricks Runtime 5.5 LTS or Databricks Runtime 5.5 LTS ML.
# MAGIC * If you are using a cluster running Databricks Runtime, you must install MLflow. See "Install a library on a cluster" ([AWS](https://docs.databricks.com/libraries/cluster-libraries.html#install-a-library-on-a-cluster)|[Azure](https://docs.microsoft.com/azure/databricks/libraries/cluster-libraries#--install-a-library-on-a-cluster)|[GCP](https://docs.gcp.databricks.com/libraries/cluster-libraries.html#install-a-library-on-a-cluster)). Select **Library Source** PyPI and enter `mlflow` in the **Package** field.
# MAGIC * If you are using a cluster running Databricks Runtime ML, MLflow is already installed.  
# MAGIC
# MAGIC ## Prerequsite
# MAGIC * This notebook uses the ElasticNet models from MLflow quickstart part 1: training and logging ([AWS](https://docs.databricks.com/applications/mlflow/tracking-ex-scikit.html#training-quickstart)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking-ex-scikit#--training-quickstart)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/tracking-ex-scikit.html#training-quickstart)).

# COMMAND ----------

# MAGIC %md ## Find and copy the run ID of the run that created the model
# MAGIC
# MAGIC Find and copy a run ID associated with an ElasticNet training run from the MLflow quickstart part 1: training and logging notebook. The run ID appears on the run details page; it is a 32-character alphanumeric string shown after the label "**Run**".  
# MAGIC
# MAGIC To navigate to the run details page for the MLflow quickstart part 1: training and logging notebook, open that notebook and click **Experiment** in the upper right corner. The Experiments sidebar displays. Do one of the following:
# MAGIC
# MAGIC * In the Experiments sidebar, click the icon at the far right of the date and time of the run. The run details page appears in a new tab. 
# MAGIC
# MAGIC * Click the square icon with the arrow to the right of **Experiment Runs**. The Experiment page displays in a new tab. This page lists all of the runs associated with this notebook. To display the run details page for a particular run, click the link in the **Start Time** column for that run. 
# MAGIC
# MAGIC For more information, see "View notebook experiment" ([AWS](https://docs.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking#view-notebook-experiment)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment)).

# COMMAND ----------

# Replace <run-id1> with the run ID you identified in the previous step.
run_id1 = "<run-id1>"
model_uri = "runs:/" + run_id1 + "/model"

# COMMAND ----------

# MAGIC %md ## Load the model as a scikit-learn model
# MAGIC Use the MLflow API to load the model from the MLflow server that was created by the run. After loading the model, you can use just like you would any scikit-learn model. 

# COMMAND ----------

import mlflow.sklearn
model = mlflow.sklearn.load_model(model_uri=model_uri)
model.coef_

# COMMAND ----------

# Import required libraries
from sklearn import datasets
import numpy as np
import pandas as pd

# Load diabetes datasets
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Create pandas DataFrame for sklearn ElasticNet linear_model
Y = np.array([y]).transpose()
d = np.concatenate((X, Y), axis=1)
cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
data = pd.DataFrame(d, columns=cols)

# COMMAND ----------

# Get a prediction for a row of the dataset
model.predict(data[0:1].drop(["progression"], axis=1))

# COMMAND ----------

# MAGIC %md ## Create a PySpark UDF and use it for batch inference
# MAGIC In this section, you use the MLflow API to create a PySpark UDF from the model you saved to MLflow. For more information, see [Export a python_function model as an Apache Spark UDF](https://mlflow.org/docs/latest/models.html#export-a-python-function-model-as-an-apache-spark-udf).  
# MAGIC
# MAGIC Saving the model as a PySpark UDF allows you to run the model to make predictions on a Spark DataFrame. 

# COMMAND ----------

# Create the PySpark UDF
import mlflow.pyfunc
pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

# COMMAND ----------

# For the purposes of this example, create a small Spark DataFrame. This is the original pandas DataFrame without the label column.
dataframe = spark.createDataFrame(data.drop(["progression"], axis=1))

# COMMAND ----------

# MAGIC %md Use the Spark function `withColumn()` to apply the PySpark UDF to the DataFrame and return a new DataFrame with a `prediction` column. 

# COMMAND ----------

from pyspark.sql.functions import struct

predicted_df = dataframe.withColumn("prediction", pyfunc_udf(struct('age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6')))
display(predicted_df)