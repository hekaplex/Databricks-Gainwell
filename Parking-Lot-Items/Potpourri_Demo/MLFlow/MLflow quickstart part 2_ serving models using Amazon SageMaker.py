# Databricks notebook source
# MAGIC %md ##MLflow quickstart part 2: serving models using Amazon SageMaker
# MAGIC
# MAGIC The [first part of this guide](https://docs.databricks.com/applications/mlflow/tracking-ex-scikit.html), **MLflow quickstart: model training and logging**, focuses on training a model and logging the training metrics, parameters, and model to the MLflow tracking server. 
# MAGIC
# MAGIC ##### NOTE: Do not use *Run All* with this notebook. It takes several minutes to deploy and update models in SageMaker, and models cannot be queried until they are active.
# MAGIC
# MAGIC This part of the guide consists of the following sections:
# MAGIC
# MAGIC #### Setup
# MAGIC * Select a model to deploy using the MLflow tracking UI
# MAGIC
# MAGIC #### Deploy a model
# MAGIC * Deploy the selected model to SageMaker using the MLflow API
# MAGIC * Check the status and health of the deployed model
# MAGIC   * Determine if the deployed model is active and ready to be queried
# MAGIC
# MAGIC #### Query the deployed model
# MAGIC * Load an input vector that the deployed model can evaluate
# MAGIC * Query the deployed model using the input
# MAGIC
# MAGIC #### Manage the deployment
# MAGIC * Update the deployed model using the MLflow API
# MAGIC * Query the updated model
# MAGIC
# MAGIC #### Clean up the deployment
# MAGIC * Delete the model deployment using the MLflow API
# MAGIC
# MAGIC As in the first part of the quickstart tutorial, this notebook uses ElasticNet models trained on the `diabetes` dataset in scikit-learn.

# COMMAND ----------

# MAGIC %md ## Prerequisites
# MAGIC
# MAGIC ElasticNet models from the MLflow quickstart notebook in [part 1 of the quickstart guide](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809225638/command/3558079809225639). /Koantek_Demo_For_Challice/MLFlow/MLflow quickstart part 1: training and logging

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Ensure you are using or create a cluster specifying: 
# MAGIC   * **Python Version:** Python 3
# MAGIC   * An attached IAM role that supports SageMaker deployment. For information about setting up a cluster IAM role for SageMaker deployment, see the [SageMaker deployment guide](https://docs.databricks.com/administration-guide/cloud-configurations/aws/sagemaker.html).
# MAGIC 1. If you are running Databricks Runtime, uncomment and run Cmd 5 to install the required libraries. If you are running Databricks Runtime for Machine Learning, you can skip this step as the required libraries are already installed. 
# MAGIC 1. Attach this notebook to the cluster.

# COMMAND ----------

# %pip install mlflow>=1.30.0

# COMMAND ----------

# MAGIC %md Choose a run ID associated with an ElasticNet training run from [part 1 of the quickstart guide](https://docs.databricks.com/applications/mlflow/tracking-ex-scikit.html). You can find a run ID and model path from the experiment run, which can be found on the MLflow UI run details page:
# MAGIC
# MAGIC ![image](https://docs.databricks.com/_static/images/mlflow/mlflow-deployment-example-run-info.png)

# COMMAND ----------

# MAGIC %md ### Set region, run ID, model URI
# MAGIC
# MAGIC **Note**: You must create a new MLflow SageMaker Deployment in each new region.

# COMMAND ----------

region = "<region>"
run_id1 = "<run-id1>"
model_uri = "runs:/" + run_id1 + "/model"

# COMMAND ----------

# MAGIC %md ### Deploy a model
# MAGIC
# MAGIC In this section, deploy the model you selected during **Setup** to SageMaker.

# COMMAND ----------

# MAGIC %md Specify a Docker image in Amazon's Elastic Container Registry (ECR). SageMaker uses this image to serve the model.  
# MAGIC To obtain the container URL, build the `mlflow-pyfunc` image and upload it to an ECR repository using the MLflow CLI: `mlflow sagemaker build-and-push-container`.

# COMMAND ----------

# MAGIC %md Define the ECR URL for the `mlflow-pyfunc` image that will be passed as an argument to MLflow's `SageMakerDeploymentClient.create()` function.

# COMMAND ----------

# Replace <ECR-URL> in the following line with the URL for your ECR docker image
# The ECR URL should have the following format: {account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}
image_ecr_url = "<ECR-URL>"

# COMMAND ----------

# MAGIC %md Use MLflow's SageMaker API to deploy your trained model to SageMaker. The `SageMakerDeploymentClient.create()` function creates all the necessary SageMaker objects, such as SageMaker Endpoints and SageMaker Models.

# COMMAND ----------

import mlflow.deployments

deployment_name = "diabetes-class"

deployment_client = mlflow.deployments.get_deploy_client("sagemaker:/" + region)
deployment_client.create_deployment(
    name=deployment_name,
    model_uri=model_uri,
    config={
      "image_url": image_ecr_url
    }
)

# COMMAND ----------

# MAGIC %md #### Using a single function, your model has now been deployed to SageMaker.

# COMMAND ----------

# MAGIC %md Check the status of your new MLflow SageMaker Deployment by running the following cell.
# MAGIC
# MAGIC **Note**: The status should be **Creating**. Wait until the status is **InService**; until then, query requests will fail.

# COMMAND ----------

deployment_info = deployment_client.get_deployment(name=deployment_name)
print(f"MLflow SageMaker Deployment status is: {deployment_info['EndpointStatus']}")

# COMMAND ----------

# MAGIC %md ### Query the deployed model

# COMMAND ----------

# MAGIC %md #### Load sample input from the `diabetes` dataset and construct a model input DataFrame

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn import datasets

# Load diabetes datasets
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Create a pandas DataFrame that serves as sample input for the deployed ElasticNet model
Y = np.array([y]).transpose()
d = np.concatenate((X, Y), axis=1)
cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
data = pd.DataFrame(d, columns=cols)
query_df = data.drop(["progression"], axis=1).iloc[[0]]

# COMMAND ----------

print(f"Using input dataframe:\n\n {query_df}")

# COMMAND ----------

# MAGIC %md #### Evaluate the sample input 
# MAGIC
# MAGIC Send the sample input to the model for inference using `SageMakerDeploymentClient.predict()`

# COMMAND ----------

prediction1 = deployment_client.predict(deployment_name, query_df)
print(f"Prediction response: {prediction1}")

# COMMAND ----------

# MAGIC %md ### Manage the deployment
# MAGIC
# MAGIC You can update the deployed model by replacing it with the output of a different run. Specify the run ID associated with a different ElasticNet training run.

# COMMAND ----------

run_id2 = "<run-id2>"
model_uri = "runs:/" + run_id2 + "/model"

# COMMAND ----------

# MAGIC %md Call `SageMakerDeploymentClient.update_deployment()` in `replace` mode. This updates the `diabetes-class` MLflow SageMaker Deployment with the model corresponding to the new run ID.

# COMMAND ----------

deployment_client.update_deployment(
  name=deployment_name,
  model_uri=model_uri,
  config={
    "image_url": image_ecr_url,
    "mode": "replace",
  }
)

# COMMAND ----------

# MAGIC %md **Note**: The MLflow SageMaker Deployment status should be **Updating**. The updated model is used once the status changes to **InService**.

# COMMAND ----------

deployment_info = deployment_client.get_deployment(name=deployment_name)
print(f"MLflow SageMaker Deployment status is: {deployment_info['EndpointStatus']}")

# COMMAND ----------

# MAGIC %md Query the updated model. You should get a different prediction.

# COMMAND ----------

prediction2 = deployment_client.predict(deployment_name, query_df)
print(f"Prediction response: {prediction2}")

# COMMAND ----------

# MAGIC %md Compare the predictions.

# COMMAND ----------

print("Run ID: {} Prediction: {}".format(run_id1, prediction1["predictions"])) 
print("Run ID: {} Prediction: {}".format(run_id2, prediction2["predictions"]))

# COMMAND ----------

# MAGIC %md ### Clean up the deployment
# MAGIC
# MAGIC When the model deployment is no longer needed, use the `SageMakerDeploymentClient.delete_deployment()` function to delete it.

# COMMAND ----------

deployment_client.delete_deployment(name=deployment_name)

# COMMAND ----------

# MAGIC %md Verify that the MLflow SageMaker Deployment has been deleted.

# COMMAND ----------

deployments = deployment_client.list_deployments()
deployment_names = [deployment['EndpointName'] for deployment in deployments]

print(f"The following MLflow SageMaker Deployments exist in {region}: {deployment_names}")