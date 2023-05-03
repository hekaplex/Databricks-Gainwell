# Databricks notebook source
# MAGIC %md
# MAGIC ### Manage model lifecycle
# MAGIC
# MAGIC This article describes how to use MLflow Model Registry as part of your machine learning workflow to manage the full lifecycle of ML models. Databricks provides a hosted version of MLflow Model Registry. Model Registry provides:
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
# MAGIC This article includes instructions for both the Model Registry UI and the Model Registry API.
# MAGIC
# MAGIC For an overview of Model Registry concepts, see [MLflow guide](https://docs.databricks.com/mlflow/index.html).
# MAGIC
# MAGIC **Create or register a model**
# MAGIC
# MAGIC You can create or register a model using the UI, or register a model using the API.
# MAGIC
# MAGIC **Create or register a model using the UI**
# MAGIC
# MAGIC There are two ways to register a model in the Model Registry. You can register an existing model that has been logged to MLflow, or you can create and register a new, empty model and then assign a previously logged model to it.
# MAGIC
# MAGIC **Register an existing logged model from a notebook**
# MAGIC
# MAGIC > 1. In the Workspace, identify the MLflow run containing the model you want to register.
# MAGIC
# MAGIC >> a. Click the **Experiment** icon ![Experiment icon](https://docs.databricks.com/_images/experiment.png) in the notebook’s right sidebar.
# MAGIC
# MAGIC ![Notebook toolbar](https://docs.databricks.com/_images/notebook-toolbar.png)
# MAGIC
# MAGIC >> b. In the Experiment Runs sidebar, click the External Link icon next to the date of the run. The MLflow Run page displays. This page shows details of the run including parameters, metrics, tags, and list of artifacts.
# MAGIC
# MAGIC > 2. In the Artifacts section, click the directory named **xxx-model**.
# MAGIC
# MAGIC ![Register model](https://docs.databricks.com/_images/register-model.png)
# MAGIC
# MAGIC
# MAGIC > 3. Click the **Register Model** button at the far right.
# MAGIC
# MAGIC > 4. In the dialog, click in the **Model** box and do one of the following:
# MAGIC
# MAGIC >> - Select **Create New Model** from the drop-down menu. The **Model Name** field appears. Enter a model name, for example scikit-learn-power-forecasting.
# MAGIC
# MAGIC >> - Select an existing model from the drop-down menu.
# MAGIC
# MAGIC ![Create new model](https://docs.databricks.com/_images/create-model.png)
# MAGIC
# MAGIC
# MAGIC > 5. Click Register.
# MAGIC
# MAGIC >> - If you selected **Create New Model**, this registers a model named scikit-learn-power-forecasting, copies the model into a secure location managed by the MLflow Model Registry, and creates a new version of the model.
# MAGIC
# MAGIC >> - If you selected an existing model, this registers a new version of the selected model.
# MAGIC
# MAGIC >> After a few moments, the **Register Model** button changes to a link to the new registered model version.
# MAGIC
# MAGIC ![Select newly created model](https://docs.databricks.com/_images/registered-model-version.png)
# MAGIC
# MAGIC
# MAGIC > 6. Click the link to open the new model version in the Model Registry UI. You can also find the model in the Model Registry by clicking ![Models Icon](https://docs.databricks.com/_images/models-icon.png) Models in the sidebar.
# MAGIC
# MAGIC **Create a new registered model and assign a logged model to it**
# MAGIC
# MAGIC You can use the Create Model button on the registered models page to create a new, empty model and then assign a logged model to it. Follow these steps:
# MAGIC
# MAGIC > 1. On the registered models page, click **Create Model**. Enter a name for the model and click Create.
# MAGIC
# MAGIC > 2. Follow Steps 1 through 3 in [Register an existing logged model from a notebook]().
# MAGIC
# MAGIC > 3. In the **Register** Model dialog, select the name of the model you created in Step 1 and click Register. This registers a model with the name you created, copies the model into a secure location managed by the MLflow Model Registry, and creates a model version: Version 1.
# MAGIC
# MAGIC > After a few moments, the MLflow Run UI replaces the Register Model button with a link to the new registered model version. You can now select the model from the **Model** drop-down list in the Register Model dialog on the **Experiment Runs** page. You can also register new versions of the model by specifying its name in API commands like [Create ModelVersion](https://mlflow.org/docs/latest/rest-api.html#create-modelversion).
# MAGIC
# MAGIC **Register a model using the API**
# MAGIC
# MAGIC There are three programmatic ways to register a model in the Model Registry. All methods copy the model into a secure location managed by the MLflow Model Registry.
# MAGIC
# MAGIC > - To log a model and register it with the specified name during an MLflow experiment, use the mlflow.<model-flavor>.log_model(...) method. If a registered model with the name doesn’t exist, the method registers a new model, creates Version 1, and returns a ModelVersion MLflow object. If a registered model with the name exists already, the method creates a new model version and returns the version object.
# MAGIC
# MAGIC Python Code:
# MAGIC   
# MAGIC `with mlflow.start_run(run_name=<run-name>) as run:`
# MAGIC   
# MAGIC `  ...`
# MAGIC   
# MAGIC `  mlflow.<model-flavor>.log_model(<model-flavor>=<model>,`
# MAGIC   
# MAGIC `    artifact_path="<model-path>",`
# MAGIC   
# MAGIC `    registered_model_name="<model-name>"`
# MAGIC   
# MAGIC `  )`
# MAGIC   
# MAGIC > - To register a model with the specified name after all your experiment runs complete and you have decided which model is most suitable to add to the registry, use the mlflow.register_model() method. For this method, you need the run ID for the mlruns:URI argument. If a registered model with the name doesn’t exist, the method registers a new model, creates Version 1, and returns a ModelVersion MLflow object. If a registered model with the name exists already, the method creates a new model version and returns the version object.
# MAGIC
# MAGIC Python Code:
# MAGIC   
# MAGIC `result=mlflow.register_model("runs:<model-path>", "<model-name>")`
# MAGIC   
# MAGIC > - To create a new registered model with the specified name, use the MLflow Client API create_registered_model() method. If the model name exists, this method throws an MLflowException.
# MAGIC
# MAGIC Python Code:
# MAGIC   
# MAGIC `client = MlflowClient()`
# MAGIC   
# MAGIC `result = client.create_registered_model("<model-name>")`
# MAGIC   
# MAGIC You can also register a model with the [Databricks Terraform provider](https://docs.databricks.com/dev-tools/terraform/index.html) and [databricks_mlflow_model](https://registry.terraform.io/providers/databricks/databricks/latest/docs/resources/mlflow_model).
# MAGIC
# MAGIC **Control access to models**
# MAGIC   
# MAGIC To learn how to control access to models registered in Model Registry, see [MLflow Model permissions](https://docs.databricks.com/security/access-control/workspace-acl.html#assign-mlflow-model-permissions).
# MAGIC
# MAGIC **Transition a model stage**
# MAGIC   
# MAGIC A model version has one of the following stages: None, Staging, Production, or Archived. The Staging stage is meant for model testing and validating, while the Production stage is for model versions that have completed the testing or review processes and have been deployed to applications for live scoring. An Archived model version is assumed to be inactive, at which point you can consider deleting it. Different versions of a model can be in different stages.
# MAGIC
# MAGIC A user with appropriate permission can transition a model version between stages. If you have permission to transition a model version to a particular stage, you can make the transition directly. If you do not have permission, you can request a stage transition and a user that has permission to transition model versions can approve, reject, or cancel the request.
# MAGIC
# MAGIC You can transition a model stage using the UI or using the API.
# MAGIC
# MAGIC **Transition a model stage using the UI**
# MAGIC   
# MAGIC Follow these instructions to transition a model’s stage.
# MAGIC
# MAGIC > 1. To display the list of available model stages and your available options, in a model version page, click the drop down next to **Stage**: and request or select a transition to another stage.
# MAGIC
# MAGIC ![Stage transition options](https://docs.databricks.com/_images/stage-options.png)
# MAGIC   
# MAGIC > 2. Enter an optional comment and click **OK**.
# MAGIC
# MAGIC **Transition a model version to the Production stage**
# MAGIC   
# MAGIC After testing and validation, you can transition or request a transition to the Production stage.
# MAGIC
# MAGIC Model Registry allows more than one version of the registered model in each stage. If you want to have only one version in Production, you can transition all versions of the model currently in Production to Archived by checking **Transition existing Production model versions to Archived**.
# MAGIC
# MAGIC **Approve, reject, or cancel a model version stage transition request**
# MAGIC   
# MAGIC A user without stage transition permission can request a stage transition. The request appears in the **Pending Requests** section in the model version page:
# MAGIC
# MAGIC ![Transition to production](https://docs.databricks.com/_images/handle-transition-request.png)
# MAGIC   
# MAGIC To approve, reject, or cancel a stage transition request, click the Approve, Reject, or Cancel link.
# MAGIC
# MAGIC The creator of a transition request can also cancel the request.
# MAGIC
# MAGIC **View model version activities**
# MAGIC   
# MAGIC To view all the transitions requested, approved, pending, and applied to a model version, go to the Activities section. This record of activities provides a lineage of the model’s lifecycle for auditing or inspection.
# MAGIC
# MAGIC **Transition a model stage using the API**
# MAGIC   
# MAGIC Users with appropriate permissions can transition a model version to a new stage.
# MAGIC
# MAGIC To update a model version stage to a new stage, use the MLflow Client API transition_model_version_stage() method:
# MAGIC
# MAGIC Python Code:
# MAGIC   
# MAGIC `  client = MlflowClient()`
# MAGIC   
# MAGIC `  client.transition_model_version_stage(`
# MAGIC   
# MAGIC `    name="<model-name>",`
# MAGIC   
# MAGIC `    version=<model-version>,`
# MAGIC   
# MAGIC `    stage="<stage>",`
# MAGIC   
# MAGIC `    description="<description>"`
# MAGIC   
# MAGIC `  )`
# MAGIC   
# MAGIC The accepted values for <stage> are: "Staging"|"staging", "Archived"|"archived", "Production"|"production", "None"|"none".
# MAGIC
# MAGIC **Use model for inference**  
# MAGIC
# MAGIC After a model is registered in Model Registry, you can automatically generate a notebook to use the model for batch or streaming inference. Alternatively, you can create an endpoint to use the model for real-time serving with [Serverless Real-Time Inference](https://docs.databricks.com/machine-learning/model-inference/serverless/serverless-real-time-inference.html).
# MAGIC
# MAGIC In the upper-right corner of the registered model page or the model version page, click ![use model button](https://docs.databricks.com/_images/use-model-for-inference.png). The Configure model inference dialog appears, which allows you to configure batch, streaming, or real-time inference.
# MAGIC
# MAGIC *Important!*
# MAGIC
# MAGIC Anaconda Inc. updated their terms of service for anaconda.org channels. Based on the new terms of service you may require a commercial license if you rely on Anaconda’s packaging and distribution. See Anaconda Commercial Edition FAQ for more information. Your use of any Anaconda channels is governed by their terms of service.
# MAGIC
# MAGIC MLflow models logged before v1.18 (Databricks Runtime 8.3 ML or earlier) were by default logged with the conda defaults channel (https://repo.anaconda.com/pkgs/) as a dependency. Because of this license change, Databricks has stopped the use of the defaults channel for models logged using MLflow v1.18 and above. The default channel logged is now conda-forge, which points at the community managed https://conda-forge.org/.
# MAGIC
# MAGIC If you logged a model before MLflow v1.18 without excluding the defaults channel from the conda environment for the model, that model may have a dependency on the defaults channel that you may not have intended. To manually confirm whether a model has this dependency, you can examine channel value in the conda.yaml file that is packaged with the logged model. For example, a model’s conda.yaml with a defaults channel dependency may look like this:
# MAGIC
# MAGIC YAML
# MAGIC Copy to clipboardCopy
# MAGIC channels:
# MAGIC - defaults
# MAGIC dependencies:
# MAGIC - python=3.8.8
# MAGIC - pip
# MAGIC - pip:
# MAGIC     - mlflow
# MAGIC     - scikit-learn==0.23.2
# MAGIC     - cloudpickle==1.6.0
# MAGIC       name: mlflow-env
# MAGIC Because Databricks can not determine whether your use of the Anaconda repository to interact with your models is permitted under your relationship with Anaconda, Databricks is not forcing its customers to make any changes. If your use of the Anaconda.com repo through the use of Databricks is permitted under Anaconda’s terms, you do not need to take any action.
# MAGIC
# MAGIC If you would like to change the channel used in a model’s environment, you can re-register the model to the model registry with a new conda.yaml. You can do this by specifying the channel in the conda_env parameter of log_model().
# MAGIC
# MAGIC For more information on the log_model() API, see the MLflow documentation for the model flavor you are working with, for example, [log_model for scikit-learn](https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model).
# MAGIC
# MAGIC For more information on conda.yaml files, see the [MLflow documentation](https://www.mlflow.org/docs/latest/models.html#additional-logged-files).
# MAGIC
# MAGIC ![Configure model inference dialog](https://docs.databricks.com/_images/configure-model-inference.png)
# MAGIC   
# MAGIC **Configure batch inference**
# MAGIC   
# MAGIC When you follow these steps to create a batch inference notebook, the notebook is saved in your user folder under the Batch-Inference folder in a folder with the model’s name. You can edit the notebook as needed.
# MAGIC
# MAGIC > 1. Click the **Batch inference** tab.
# MAGIC
# MAGIC > 2. From the **Model version** drop-down, select the model version to use. The first two items in the drop-down are the current Production and Staging version of the model (if they exist). When you select one of these options, the notebook automatically uses the Production or Staging version as of the time it is run. You do not need to update the notebook as you continue to develop the model.
# MAGIC
# MAGIC > 3. Click the **Browse** button next to **Input table**. The **Select input data** dialog appears. If necessary, you can change the cluster in the **Compute** drop-down.
# MAGIC
# MAGIC Note
# MAGIC
# MAGIC For Unity Catalog enabled workspaces, the **Select input data** dialog allows you to select from three levels, <catalog_name>.<database_name>.<table_name>.
# MAGIC
# MAGIC > 4. Select the table containing the input data for the model, and click **Select**. The generated notebook automatically imports this data and sends it to the model. You can edit the generated notebook if the data requires any transformations before it is input to the model.
# MAGIC
# MAGIC > 5. Predictions are saved in a folder in the directory *dbfs:/FileStore/batch-inference*. By default, predictions are saved in a folder with the same name as the model. Each run of the generated notebook writes a new file to this directory with the timestamp appended to the name. You can also choose not to include the timestamp and to overwrite the file with subsequent runs of the notebook; instructions are provided in the generated notebook.
# MAGIC
# MAGIC > You can change the folder where the predictions are saved by typing a new folder name into the Output table location field or by clicking the folder icon to browse the directory and select a different folder.
# MAGIC
# MAGIC > To save predictions to a location in Unity Catalog, you must edit the notebook. For an example notebook that shows how to train a machine-learning model that uses data in Unity Catalog and write the results back to Unity Catalog, see [Python ML model training with Unity Catalog data](https://docs.databricks.com/data-governance/unity-catalog/machine-learning.html).
# MAGIC
# MAGIC **Configure streaming inference using Delta Live Tables**
# MAGIC   
# MAGIC When you follow these steps to create a streaming inference notebook, the notebook is saved in your user folder under the DLT-Inference folder in a folder with the model’s name. You can edit the notebook as needed.
# MAGIC
# MAGIC > 1. Click the **Streaming (Delta Live Tables)** tab.
# MAGIC
# MAGIC > 2. From the **Model version** drop-down, select the model version to use. The first two items in the drop-down are the current Production and Staging version of the model (if they exist). When you select one of these options, the notebook automatically uses the Production or Staging version as of the time it is run. You do not need to update the notebook as you continue to develop the model.
# MAGIC
# MAGIC > 3. Click the **Browse** button next to **Input table**. The **Select input data** dialog appears. If necessary, you can change the cluster in the **Compute** drop-down.
# MAGIC
# MAGIC Note
# MAGIC
# MAGIC For Unity Catalog enabled workspaces, the **Select input data** dialog allows you to select from three levels, <catalog_name>.<database_name>.<table_name>.
# MAGIC
# MAGIC > 4. Select the table containing the input data for the model, and click Select. The generated notebook creates a data transform that uses the input table as a source and integrates the MLflow [PySpark inference UDF](https://mlflow.org/docs/latest/models.html#export-a-python-function-model-as-an-apache-spark-udf) to perform model predictions. You can edit the generated notebook if the data requires any additional transformations before or after the model is applied.
# MAGIC
# MAGIC > 5. Provide the output Delta Live Table name. The notebook creates a live table with the given name and uses it to store the model predictions. You can modify the generated notebook to customize the target dataset as needed - for example: define a streaming live table as output, add schema information or data quality constraints.
# MAGIC
# MAGIC > 6. You can then either [create a new](https://docs.databricks.com/workflows/delta-live-tables/delta-live-tables-ui.html#create-a-pipeline) Delta Live Tables pipeline with this notebook or [add it](https://docs.databricks.com/workflows/delta-live-tables/delta-live-tables-ui.html#create-a-pipeline) to an existing pipeline as an additional notebook library.
# MAGIC
# MAGIC **Configure real-time inference**
# MAGIC   
# MAGIC [Serverless Real-Time Inference](https://docs.databricks.com/machine-learning/model-inference/serverless/serverless-real-time-inference.html) exposes your MLflow machine learning models as scalable REST API endpoints. To create a serverless endpoint for model serving, see Create and manage [Serverless Real-Time Inference endpoints](https://docs.databricks.com/machine-learning/model-inference/serverless/create-manage-serverless-endpoints.html).
# MAGIC
# MAGIC
# MAGIC **Compare model versions**
# MAGIC   
# MAGIC You can compare model versions in Model Registry.
# MAGIC
# MAGIC > 1. On the **registered model page**, select two or more model versions by clicking in the checkbox to the left of the model version.
# MAGIC
# MAGIC > 2. Click **Compare**.
# MAGIC
# MAGIC > 3. The Comparing <N> Versions screen appears, showing a table that compares the parameters, schema, and metrics of the selected model versions. At the bottom of the screen, you can select the type of plot (scatter, contour, or parallel coordinates) and the parameters or metrics to plot.
# MAGIC
# MAGIC **Control notification preferences**
# MAGIC   
# MAGIC You can configure Model Registry to notify you by email about activity on registered models and model versions that you specify.
# MAGIC
# MAGIC On the registered model page, the **Notify me about** menu shows three options:
# MAGIC
# MAGIC ![Email notifications menu](https://docs.databricks.com/_images/email-notifications-menu.png)
# MAGIC   
# MAGIC > - **All new activity**: Send email notifications about all activity on all model versions of this model. If you created the registered model, this setting is the default.
# MAGIC
# MAGIC > - **Activity on versions I follow: Send email notifications only about model versions you follow. With this selection, you receive notifications for all model versions that you follow**; you cannot turn off notifications for a specific model version.
# MAGIC
# MAGIC > - **Mute notifications**: Do not send email notifications about activity on this registered model.
# MAGIC
# MAGIC The following events trigger an email notification:
# MAGIC
# MAGIC > -  Creation of a new model version
# MAGIC
# MAGIC > - Request for a stage transition
# MAGIC
# MAGIC > - Stage transition
# MAGIC
# MAGIC > - New comments
# MAGIC
# MAGIC You are automatically subscribed to model notifications when you do any of the following:
# MAGIC
# MAGIC > - Comment on that model version
# MAGIC
# MAGIC > - Transition a model version’s stage
# MAGIC
# MAGIC > - Make a transition request for the model’s stage
# MAGIC
# MAGIC To see if you are following a model version, look at the Follow Status field on the **model version page**, or at the table of model versions on the **registered model page**.
# MAGIC
# MAGIC **Turn off all email notifications**
# MAGIC   
# MAGIC You can turn off email notifications in the Model Registry Settings tab of the User Settings menu:
# MAGIC
# MAGIC > 1. Click ![User Settings Icon](https://docs.databricks.com/_images/user-settings-icon.png) Settings in the lower left corner of your Databricks workspace.
# MAGIC
# MAGIC > 2. Click User Settings.
# MAGIC
# MAGIC > 3. Go to the Email preferences tab.
# MAGIC
# MAGIC > 4. Turn off Model Registry email notifications.
# MAGIC
# MAGIC An admin can turn off email notifications for the entire organization in the Admin Console.
# MAGIC
# MAGIC **Maximum number of emails sent**
# MAGIC   
# MAGIC Model Registry limits the number of emails sent to each user per day per activity. For example, if you receive 20 emails in one day about new model versions created for a registered model, Model Registry sends an email noting that the daily limit has been reached, and no additional emails about that event are sent until the next day.
# MAGIC
# MAGIC To increase the limit of the number of emails allowed, contact your Databricks representative.
# MAGIC
# MAGIC **Webhooks**
# MAGIC   
# MAGIC
# MAGIC [Webhooks](https://docs.databricks.com/mlflow/model-registry-webhooks.html) enable you to listen for Model Registry events so your integrations can automatically trigger actions. You can use webhooks to automate and integrate your machine learning pipeline with existing CI/CD tools and workflows. For example, you can trigger CI builds when a new model version is created or notify your team members through Slack each time a model transition to production is requested.
# MAGIC
# MAGIC **Annotate a model or model version**
# MAGIC   
# MAGIC You can provide information about a model or model version by annotating it. For example, you may want to include an overview of the problem or information about the methodology and algorithm used.
# MAGIC
# MAGIC Annotate a model or model version using the UI
# MAGIC The Databricks UI provides several ways to annotate models and model versions. You can add text information using a description or comments, and you can add **searchable key-value tags**. Descriptions and tags are available for models and model versions; comments are only available for model versions.
# MAGIC
# MAGIC > - Descriptions are intended to provide information about the model.
# MAGIC
# MAGIC > - Comments provide a way to maintain an ongoing discussion about activities on a model version.
# MAGIC
# MAGIC > - Tags let you customize model metadata to make it easier to find specific models.
# MAGIC
# MAGIC **Add or update the description for a model or model version**
# MAGIC   
# MAGIC > 1. From the **registered model** or **model version** page, click **Edit** next to **Description**. An edit window appears.
# MAGIC
# MAGIC > 2. Enter or edit the description in the edit window.
# MAGIC
# MAGIC > 3. Click **Save** to save your changes or **Cancel** to close the window.
# MAGIC
# MAGIC > If you entered a description of a model version, the description appears in the Description column in the table on the **registered model page**. The column displays a maximum of 32 characters or one line of text, whichever is shorter.
# MAGIC
# MAGIC **Add comments for a model version**
# MAGIC   
# MAGIC > 1. Scroll down the **model version** page and click the down arrow next to **Activities**.
# MAGIC
# MAGIC > 2. Type your comment in the edit window and click **Add Comment**.
# MAGIC
# MAGIC **Add tags for a model or model version**
# MAGIC   
# MAGIC > 1. From the **registered model** or **model version** page, click ![Tag icon](https://docs.databricks.com/_images/tags2.png) if it is not already open. The tags table appears.
# MAGIC
# MAGIC ![tag table](https://docs.databricks.com/_images/tags-open.png)
# MAGIC   
# MAGIC > 2. Click in the **Name** and **Value** fields and type the key and value for your tag.
# MAGIC
# MAGIC > 3. Click **Add**.
# MAGIC
# MAGIC ![add tag](https://docs.databricks.com/_images/tag-add.png)
# MAGIC   
# MAGIC **Edit or delete tags for a model or model version**
# MAGIC   
# MAGIC To edit or delete an existing tag, use the icons in the Actions column.
# MAGIC
# MAGIC ![tag actions](https://docs.databricks.com/_images/tag-edit-or-delete.png)
# MAGIC   
# MAGIC **Annotate a model version using the API**
# MAGIC   
# MAGIC To update a model version description, use the MLflow Client API *update_model_version()* method:
# MAGIC
# MAGIC Python Code:
# MAGIC   
# MAGIC `client = MlflowClient()`
# MAGIC   
# MAGIC `client.update_model_version(`
# MAGIC
# MAGIC `  name="<model-name>",`
# MAGIC
# MAGIC `  version=<model-version>,`
# MAGIC
# MAGIC `  description="<description>"`
# MAGIC
# MAGIC `)`
# MAGIC   
# MAGIC To set or update a tag for a registered model or model version, use the MLflow Client API [`set_registered_model_tag()`](https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.set_registered_model_tag)) or [`set_model_version_tag()`](https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.set_model_version_tag) method:
# MAGIC
# MAGIC Python Code:
# MAGIC   
# MAGIC `client = MlflowClient()`
# MAGIC
# MAGIC `client.set_registered_model_tag()(`
# MAGIC
# MAGIC `  name="<model-name>",`
# MAGIC
# MAGIC `  key="<key-value>",`
# MAGIC
# MAGIC `  tag="<tag-value>"`
# MAGIC   
# MAGIC `)`
# MAGIC   
# MAGIC Python Code:
# MAGIC   
# MAGIC `client = MlflowClient()`
# MAGIC
# MAGIC `client.set_model_version_tag()(`
# MAGIC
# MAGIC `  name="<model-name>",`
# MAGIC
# MAGIC `  version=<model-version>,`
# MAGIC
# MAGIC `  key="<key-value>",`
# MAGIC
# MAGIC `  tag="<tag-value>"`
# MAGIC
# MAGIC `)`
# MAGIC   
# MAGIC **Rename a model (API only)**
# MAGIC   
# MAGIC To rename a registered model, use the MLflow Client API *rename_registered_model()* method:
# MAGIC
# MAGIC Python Code:
# MAGIC   
# MAGIC `client=MlflowClient()`
# MAGIC
# MAGIC `client.rename_registered_model("<model-name>", "<new-model-name>")`
# MAGIC   
# MAGIC Note
# MAGIC
# MAGIC You can rename a registered model only if it has no versions, or all versions are in the None or Archived stage.
# MAGIC
# MAGIC **Search for a model**
# MAGIC   
# MAGIC All registered models live in the MLflow Model Registry. You can search for models using the UI or the API.
# MAGIC
# MAGIC Note
# MAGIC
# MAGIC When you search for a model, only models for which you have at least “Can Read” permissions are returned.
# MAGIC
# MAGIC **Search for a model using the UI**
# MAGIC   
# MAGIC To display all registered models, click ![Models Icon](https://docs.databricks.com/_images/models-icon.png) Models in the sidebar.
# MAGIC
# MAGIC To search for a specific model, enter text in the search box. You can enter the name of a model or any part of the name:
# MAGIC
# MAGIC ![Registered models search](https://docs.databricks.com/_images/registered-models-search.png)
# MAGIC   
# MAGIC You can also search on tags. Enter tags in this format: tags.<key>=<value>. To search for multiple tags, use the AND operator.
# MAGIC
# MAGIC ![Tag-based search](https://docs.databricks.com/_images/search-with-tags.png)
# MAGIC   
# MAGIC You can search on both the model name and tags using the [MLflow search syntax](https://www.mlflow.org/docs/latest/search-syntax.html). For example:
# MAGIC
# MAGIC ![Name and tag-based search](https://docs.databricks.com/_images/model-search-name-and-tag.png)
# MAGIC   
# MAGIC **Search for a model using the API**
# MAGIC   
# MAGIC You can search for registered models in the Model Registry with the MLflow Client API method [search_registered_models()](https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.search_registered_models)
# MAGIC
# MAGIC If you have set tags on your models, you can also search by those tags with search_registered_models().
# MAGIC
# MAGIC Python Code:
# MAGIC   
# MAGIC `print(f"Find registered models with a specific tag value")`
# MAGIC
# MAGIC `for m in client.search_registered_models(f"tags.`<key-value>`='<tag-value>'"):`
# MAGIC
# MAGIC `  pprint(dict(m), indent=4)`
# MAGIC   
# MAGIC You can also search for a specific model name and list its version details using MLflow Client API search_model_versions() method:
# MAGIC
# MAGIC Python Code:
# MAGIC   
# MAGIC `from pprint import pprint`
# MAGIC   
# MAGIC `client=MlflowClient()`
# MAGIC   
# MAGIC `[pprint(mv) for mv in client.search_model_versions("name='<model-name>'")]`
# MAGIC   
# MAGIC This outputs:
# MAGIC
# MAGIC Console:
# MAGIC   
# MAGIC ![Copy to clipboardCopy](https://dbc-00646cc6-55ee.cloud.databricks.com/files/console.PNG?o=512145377436569)
# MAGIC   
# MAGIC **Delete a model or model version**
# MAGIC   
# MAGIC You can delete a model using the UI or the API.
# MAGIC
# MAGIC **Delete a model version or model using the UI**
# MAGIC   
# MAGIC *Warning*
# MAGIC
# MAGIC You cannot undo this action. You can transition a model version to the Archived stage rather than deleting it from the registry. When you delete a model, all model artifacts stored by the Model Registry and all the metadata associated with the registered model are deleted.
# MAGIC
# MAGIC *Note*
# MAGIC
# MAGIC You can only delete models and model versions in the None or Archived stage. If a registered model has versions in the Staging or Production stage, you must transition them to either the None or Archived stage before deleting the model.
# MAGIC
# MAGIC To delete a model version:
# MAGIC
# MAGIC > 1. Click ![Models Icon](https://docs.databricks.com/_images/models-icon.png) Models in the sidebar.
# MAGIC
# MAGIC > 2. Click a model name.
# MAGIC
# MAGIC > 3. Click a model version.
# MAGIC
# MAGIC > 4. Click ![Delete model version](https://docs.databricks.com/_images/three-button-icon.png) at the upper right corner of the screen and select Delete from the drop-down menu.
# MAGIC
# MAGIC To delete a model:
# MAGIC
# MAGIC > 1. Click ![Models Icon](https://docs.databricks.com/_images/models-icon.png) Models in the sidebar.
# MAGIC
# MAGIC > 2. Click a model name.
# MAGIC
# MAGIC > 3. Click ![Delete model](https://docs.databricks.com/_images/three-button-icon.png) at the upper right corner of the screen and select Delete from the drop-down menu.
# MAGIC
# MAGIC **Delete a model version or model using the API**
# MAGIC   
# MAGIC *Warning*
# MAGIC
# MAGIC You cannot undo this action. You can transition a model version to the Archived stage rather than deleting it from the registry. When you delete a model, all model artifacts stored by the Model Registry and all the metadata associated with the registered model are deleted.
# MAGIC
# MAGIC *Note*
# MAGIC
# MAGIC You can only delete models and model versions in the None or Archived stage. If a registered model has versions in the Staging or Production stage, you must transition them to either the None or Archived stage before deleting the model.
# MAGIC
# MAGIC **Delete a model version**
# MAGIC   
# MAGIC To delete a model version, use the MLflow Client API delete_model_version() method:
# MAGIC
# MAGIC Python Code:
# MAGIC   
# MAGIC `# Delete versions 1,2, and 3 of the model`
# MAGIC
# MAGIC `client = MlflowClient()`
# MAGIC
# MAGIC `versions=[1, 2, 3]`
# MAGIC
# MAGIC `for version in versions:`
# MAGIC   ``
# MAGIC `  client.delete_model_version(name="<model-name>", version=version)`
# MAGIC
# MAGIC   
# MAGIC **Delete a model**
# MAGIC   
# MAGIC To delete a model, use the MLflow Client API *delete_registered_model()* method:
# MAGIC
# MAGIC Python Code:
# MAGIC   
# MAGIC `client = MlflowClient()`
# MAGIC   
# MAGIC `client.delete_registered_model(name="<model-name>")`
# MAGIC   
# MAGIC **Share models across workspaces**
# MAGIC   
# MAGIC Databricks supports [sharing models across multiple workspaces](https://docs.databricks.com/machine-learning/manage-model-lifecycle/multiple-workspaces.html). For example, you can develop and log a model in your own workspace and then access it from another workspace using a remote model registry. This is useful when multiple teams share access to models. You can create multiple workspaces and use and manage models across these environments.
# MAGIC
# MAGIC **Copy MLflow objects between workspaces**
# MAGIC   
# MAGIC To import or export MLflow objects to or from your Databricks workspace, you can use the community-driven open source project [MLflow Export-Import](https://github.com/mlflow/mlflow-export-import#why-use-mlflow-export-import) to migrate MLflow experiments, models, and runs between workspaces.
# MAGIC
# MAGIC With these tools, you can:
# MAGIC
# MAGIC > - Share and collaborate with other data scientists in the same or another tracking server. For example, you can clone an experiment from another user into your workspace.
# MAGIC
# MAGIC > - Copy a model from one workspace to another, such as from a development to a production workspace.
# MAGIC
# MAGIC > - Copy MLflow experiments and runs from your local tracking server to your Databricks workspace.
# MAGIC
# MAGIC > - Back up mission critical experiments and models to another Databricks workspace.
# MAGIC
# MAGIC **Example**
# MAGIC   
# MAGIC This example illustrates how to use the Model Registry to build a machine learning application:
# MAGIC   
# MAGIC   [MLflow Model Registry example](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809226043/command/3558079809226044) /Koantek_Demo_For_Challice/MLFlow/MLflow Model Registry example.