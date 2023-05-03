# Databricks notebook source
# MAGIC %md
# MAGIC ### MLflow Model Registry on Databricks
# MAGIC
# MAGIC [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html) is a centralized model repository and a UI and set of APIs that enable you to manage the full lifecycle of MLflow Models. Model Registry provides:
# MAGIC
# MAGIC > - Chronological model lineage (which MLflow experiment and run produced the model at a given time).
# MAGIC
# MAGIC > - Model serving with [Serverless Real-Time Inference](https://docs.databricks.com/machine-learning/model-inference/serverless/serverless-real-time-inference.html).
# MAGIC
# MAGIC > - Model versioning.
# MAGIC
# MAGIC > - Stage transitions (for example, from staging to production or archived).
# MAGIC
# MAGIC > - [Webhooks](https://docs.databricks.com/mlflow/model-registry-webhooks.html) so you can automatically trigger actions based on registry events.
# MAGIC
# MAGIC > - Email notifications of model events.
# MAGIC
# MAGIC You can also create and view model descriptions and leave comments.
# MAGIC
# MAGIC You can work with the model registry using either the Model Registry UI or the Model Registry API. This page presents conceptual information and also includes an example notebook illustrating what you can do with the Model Registry.
# MAGIC
# MAGIC For instructions on how to use the Model Registry, see [Manage model lifecycle](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809226041/command/3558079809226042) /Koantek_Demo_For_Challice/MLFlow/Manage_model_lifecycle.
# MAGIC
# MAGIC **Model Registry concepts**
# MAGIC
# MAGIC > - **Model**: An MLflow Model logged from an experiment or run that is logged with one of the model flavor’s mlflow.<model-flavor>.log_model methods. Once logged, you can register the model with the Model Registry.
# MAGIC
# MAGIC > - **Registered model**: An MLflow Model that has been registered with the Model Registry. The registered model has a unique name, versions, model lineage, and other metadata.
# MAGIC
# MAGIC > - **Model version**: A version of a registered model. When a new model is added to the Model Registry, it is added as Version 1. Each model registered to the same model name increments the version number.
# MAGIC
# MAGIC > - **Model stage**: A model version can be assigned one or more stages. MLflow provides predefined stages for the common use-cases None, Staging, Production, and Archived. With the appropriate permission you can transition a model version between stages or you can request a model stage transition.
# MAGIC
# MAGIC > - **Description**: You can annotate a model’s intent, including description and any relevant information useful for the team such as algorithm description, dataset employed, or methodology.
# MAGIC
# MAGIC > - **Activities**: Each registered model’s activities—such as request for stage transition—is recorded. The trace of activities provides lineage and auditability of the model’s evolution, from experimentation to staged versions to production.
# MAGIC
# MAGIC **Registered models page**
# MAGIC   
# MAGIC The registered models page displays when you click ![Models Icon](https://docs.databricks.com/_images/models-icon.png) Models in the sidebar. This page shows all of the models in the registry with their current stage, last modified time, tags, and serving status. To display only models that have serving enabled, select Serving enabled from the All models dropdown.
# MAGIC
# MAGIC You can create a new model from this page.
# MAGIC
# MAGIC Also from this page, workspace administrators can [set permissions for all models in the model registry](https://docs.databricks.com/security/access-control/workspace-acl.html#configure-permissions-for-all-mlflow-models-in-model-registry).
# MAGIC
# MAGIC ![Registered models](https://docs.databricks.com/_images/registered-models.png)
# MAGIC   
# MAGIC **Registered model page**
# MAGIC   
# MAGIC To display the registered model page for a model, click a model name in the registered models page. The registered model page shows information about the selected model and a table with information about each version of the model. From this page, you can also:
# MAGIC
# MAGIC > - Set up model serving with [Serverless Real-Time Inference](https://docs.databricks.com/machine-learning/model-inference/serverless/serverless-real-time-inference.html).
# MAGIC
# MAGIC > - Automatically generate a notebook to use the model for inference.
# MAGIC
# MAGIC > - Configure email notifications.
# MAGIC
# MAGIC > - Compare model versions.
# MAGIC
# MAGIC > - Set permissions for the model.
# MAGIC
# MAGIC > - Delete a model.
# MAGIC
# MAGIC ![Registered model](https://docs.databricks.com/_images/registered-model.png)
# MAGIC   
# MAGIC **Model version page**
# MAGIC   
# MAGIC To view the model version page, do one of the following:
# MAGIC
# MAGIC > - Click a version name in the Latest Version column on the registered models page.
# MAGIC
# MAGIC > - Click a version name in the Version column in the registered model page.
# MAGIC
# MAGIC This page displays information about a specific version of a registered model and also provides a link to the source run (the version of the notebook that was run to create the model). From this page, you can also:
# MAGIC
# MAGIC > - Automatically generate a notebook to use the model for inference.
# MAGIC
# MAGIC > - Delete a model.
# MAGIC
# MAGIC ![Model version](https://docs.databricks.com/_images/model-version.png)
# MAGIC   
# MAGIC **Example**
# MAGIC   
# MAGIC For an example that illustrates how to use the Model Registry to build a machine learning application that forecasts the daily power output of a wind farm, see:
# MAGIC
# MAGIC [MLflow Model Registry example](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809226043/command/3558079809226044) /Koantek_Demo_For_Challice/MLFlow/MLflow Model Registry example