# Databricks notebook source
# MAGIC %md ##MLflow: Deploying PySpark models saved as MLeap to SageMaker
# MAGIC
# MAGIC This notebook is part 2 of the MLflow MLeap example. The [first part](https://docs.databricks.com/applications/mlflow/tracking-ex-pyspark.html#training-pyspark), **MLflow Deployment: Train PySpark Model and Log in MLeap Format**, focuses on training a PySpark model and logs the training metrics, parameters, and model in MLeap format to the MLflow tracking server. 
# MAGIC
# MAGIC ##### Note: We do not recommend using *Run All* because it takes several minutes to deploy and update models in SageMaker; models cannot be queried until they are active.
# MAGIC
# MAGIC The notebook contains the following sections:
# MAGIC
# MAGIC #### Setup
# MAGIC * Launch a Python 3 cluster configured with an IAM role for SageMaker deployment
# MAGIC * Install the MLeap Scala libraries
# MAGIC * Install MLflow and boto3.
# MAGIC
# MAGIC #### Deploy the model to SageMaker
# MAGIC * Specify a Docker image URI for deployment
# MAGIC * Use MLflow to deploy the model to SageMaker
# MAGIC * Check the status of the deployed model
# MAGIC   * Determine if the deployed model is active and ready to be queried
# MAGIC
# MAGIC #### Query the deployed model
# MAGIC * Construct a query using test data
# MAGIC * Evaluate the query using the deployed model
# MAGIC
# MAGIC #### Clean up the deployment
# MAGIC * Delete the model deployment using the MLflow API
# MAGIC * Confirm that the deployment was terminated

# COMMAND ----------

# MAGIC %md ## Setup

# COMMAND ----------

# MAGIC %md ### Create a cluster and install MLflow and MLeap on your cluster
# MAGIC
# MAGIC 1. Create a GPU-enabled cluster with the following:
# MAGIC     - **Python Version:** Python 3
# MAGIC     - An attached IAM role that supports SageMaker deployment. For information about setting up a cluster IAM role for SageMaker deployment, see the [SageMaker deployment guide](https://docs.databricks.com/administration-guide/cloud-configurations/aws/sagemaker.html).
# MAGIC 1. Install required libraries.
# MAGIC    1. Create library with Source **Maven Coordinate** and the fully-qualified Maven artifact coordinate: 
# MAGIC       * `ml.combust.mleap:mleap-spark_2.11:0.13.0`
# MAGIC    1. Install the libraries into the cluster.
# MAGIC 1. If you are running Databricks Runtime, run Cmd 4 to install mlflow. If you are using Databricks Runtime ML, you can skip this step as the required libraries are already installed. 
# MAGIC 1. Attach this notebook to the cluster.

# COMMAND ----------

dbutils.library.installPyPI("mlflow", extras="extras")
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ### Load pipeline training data
# MAGIC
# MAGIC Load data that will be used to train the PySpark Pipeline model. This model uses the [20 Newsgroups dataset](http://kdd.ics.uci.edu/databases/20newsgroups/20newsgroups.html) which consists of articles from 20 Usenet newsgroups.

# COMMAND ----------

df = spark.read.parquet("/databricks-datasets/news20.binary/data-001/training").select("text", "topic")
df.cache()
display(df)

# COMMAND ----------

# MAGIC %md Specify the run ID associated with an PySpark training run from [part 1](https://docs.databricks.com/applications/mlflow/mlflow-training.html). You can find a run ID and model path from the experiment run, which can be found on the run details page:
# MAGIC
# MAGIC   ![image](https://docs.databricks.com/_static/images/mlflow/mlflow-deployment-example-run-info.png)

# COMMAND ----------

# MAGIC %md ### Set region, run ID, model URI
# MAGIC
# MAGIC **Note**: You must create a new SageMaker endpoint for each new region.

# COMMAND ----------

region = "<region>"
model_uri = "runs:/" + <run-id> + "/model"

# COMMAND ----------

# MAGIC %md ## Deploy the model to SageMaker

# COMMAND ----------

# MAGIC %md Specify a Docker image in Amazon's Elastic Container Registry (ECR) that will be used by SageMaker to serve the model. There are two ways to obtain the container URL:
# MAGIC
# MAGIC * [Option 1] You can build your own `mlflow-pyfunc` image and upload it to an ECR repository using the MLflow CLI: `mlflow sagemaker build-and-push-container`.
# MAGIC * [Option 2] Contact your Databricks representative for an `mlflow-pyfunc` image URL in ECR.

# COMMAND ----------

# MAGIC %md Define the ECR URL for the `mlflow-pyfunc` image that will be passed as an argument to MLflow's `deploy` function.

# COMMAND ----------

# the ECR URL should look like: {account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}
image_ecr_url = "<ECR-URL>"

# COMMAND ----------

# MAGIC %md ### Use MLflow to deploy the model to SageMaker
# MAGIC
# MAGIC Using MLflow's SageMaker deployment API, deploy the trained model to SageMaker.

# COMMAND ----------

import mlflow.sagemaker as mfs

app_name = "mleap-app"
mfs.deploy(app_name=app_name, model_uri=model_uri, image_url=image_ecr_url, mode="replace", flavor="mleap", region_name=region)

# COMMAND ----------

# MAGIC %md ### Check the status of the deployed model
# MAGIC
# MAGIC Check the status of the new SageMaker endpoint using a simple function.
# MAGIC
# MAGIC **Note**: The application status should be **Creating**. Wait until the status is **InService** before continuing; until then, query requests will fail. 

# COMMAND ----------

import boto3

def check_status(app_name):
  sage_client = boto3.client('sagemaker', region_name=region)
  endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
  endpoint_status = endpoint_description["EndpointStatus"]
  return endpoint_status

print("Application status is: {}".format(check_status(app_name)))

# COMMAND ----------

# MAGIC %md ## Query the deployed model

# COMMAND ----------

# MAGIC %md ### Construct a query using test data
# MAGIC
# MAGIC Load data from the [20 Newsgroups dataset](http://kdd.ics.uci.edu/databases/20newsgroups/20newsgroups.html) and construct a query DataFrame for the deployed model to evaluate.

# COMMAND ----------

# Load some test data
test_data = spark.read.parquet("/databricks-datasets/news20.binary/data-001/test").select("text", "topic")
query_df = test_data.limit(10)
display(query_df)

# COMMAND ----------

# MAGIC %md ### Evaluate the query using the deployed model
# MAGIC
# MAGIC Transform the query dataframe into JSON format and evaluate it by posting the JSON to the deployed model.
# MAGIC
# MAGIC **Note**: Deployed MLeap models only process JSON-serialized Pandas dataframes with the [**split**](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_json.html) orientation. You can convert a Spark DataFrame to this format as follows:
# MAGIC ```
# MAGIC model_input_json = spark_dataframe.toPandas().to_json(orient='split')
# MAGIC ```

# COMMAND ----------

import json

def query_endpoint(app_name, input_json):
  client = boto3.session.Session().client("sagemaker-runtime", region)
  response = client.invoke_endpoint(
      EndpointName=app_name,
      Body=input_json,
      ContentType='application/json',
  )
  preds = response['Body'].read().decode("ascii")
  preds = json.loads(preds)
  print("Received response: {}".format(preds))
  return preds

print("Sending batch prediction request with input: {input_df}".format(input_df=query_df))

# Convert the test dataframe into a JSON-serialized Pandas dataframe
input_json = query_df.toPandas().to_json(orient="split")

# Evaluate the input by posting it to the deployed model
prediction = query_endpoint(app_name=app_name, input_json=input_json)

# COMMAND ----------

# MAGIC %md ## Clean up the deployment
# MAGIC
# MAGIC Finally, terminate the deployment using MLflow and confirm that the deployment has been terminated.

# COMMAND ----------

# MAGIC %md ### Delete the deployment using MLflow

# COMMAND ----------

mfs.delete(app_name=app_name, region_name=region)

# COMMAND ----------

# MAGIC %md ### Confirm that the deployment was terminated
# MAGIC
# MAGIC By executing the following function, you should see that the SageMaker endpoints associated with the application have been removed.

# COMMAND ----------

def get_active_endpoints(app_name):
  sage_client = boto3.client('sagemaker', region_name=region)
  app_endpoints = sage_client.list_endpoints(NameContains=app_name)["Endpoints"]
  return list(filter(lambda en : en == app_name, [str(endpoint["EndpointName"]) for endpoint in app_endpoints]))
  
print("The following endpoints exist for the `{an}` application: {eps}".format(an=app_name, eps=get_active_endpoints(app_name)))