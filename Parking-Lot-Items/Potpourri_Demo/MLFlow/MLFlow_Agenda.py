# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC [MLflow](https://www.mlflow.org/) is an open source platform for managing the end-to-end machine learning lifecycle. It has the following primary components:
# MAGIC
# MAGIC > - **Tracking**: Allows you to track experiments to record and compare parameters and results.
# MAGIC
# MAGIC > - **Models**: Allow you to manage and deploy models from a variety of ML libraries to a variety of model serving and inference platforms.
# MAGIC
# MAGIC > - **Projects**: Allow you to package ML code in a reusable, reproducible form to share with other data scientists or transfer to production.
# MAGIC
# MAGIC > - **Model Registry**: Allows you to centralize a model store for managing models’ full lifecycle stage transitions: from staging to production, with capabilities for versioning and annotating.
# MAGIC
# MAGIC > - **Model Serving**: Allows you to host MLflow Models as REST endpoints.
# MAGIC
# MAGIC Databricks provides a fully managed and hosted version of MLflow integrated with enterprise security features, high availability, and other Databricks workspace features such as experiment and run management and notebook revision capture. MLflow on Databricks offers an integrated experience for tracking and securing machine learning model training runs and running machine learning projects.
# MAGIC
# MAGIC MLflow supports Java, Python, R, and REST APIs.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get started with MLflow experiments

# COMMAND ----------

# MAGIC %md
# MAGIC **MLflow tracking with experiments and runs**
# MAGIC
# MAGIC MLflow tracking is based on two concepts, experiments and runs:
# MAGIC
# MAGIC > - An MLflow experiment is the primary unit of organization and access control for MLflow runs; all MLflow runs belong to an experiment. Experiments let you visualize, search for, and compare runs, as well as download run artifacts and metadata for analysis in other tools.
# MAGIC
# MAGIC > - An MLflow run corresponds to a single execution of model code. Each run records the following information:
# MAGIC
# MAGIC >> - Source: Name of the notebook that launched the run or the project name and entry point for the run.
# MAGIC
# MAGIC >> - Version: Notebook revision if run from a notebook in a Databricks workspace, or Git commit hash if run from Databricks Repos or from an MLflow Project.
# MAGIC
# MAGIC >> - Start & end time: Start and end time of the run.
# MAGIC
# MAGIC >> - Parameters: Model parameters saved as key-value pairs. Both keys and values are strings.
# MAGIC
# MAGIC >> - Metrics: Model evaluation metrics saved as key-value pairs. The value is numeric. Each metric can be updated throughout the course of the run (for example, to track how your model’s loss function is converging), and MLflow records and lets you visualize the metric’s history.
# MAGIC
# MAGIC >> - Tags: Run metadata saved as key-value pairs. You can update tags during and after a run completes. Both keys and values are strings.
# MAGIC
# MAGIC >> - Artifacts: Output files in any format. For example, you can record images, models (for example, a pickled scikit-learn model), and data files (for example, a Parquet file) as an artifact.
# MAGIC
# MAGIC The MLflow Tracking API logs parameters, metrics, tags, and artifacts from a model run. The Tracking API communicates with an MLflow tracking server. When you use Databricks, a Databricks-hosted tracking server logs the data. The hosted MLflow tracking server has Python, Java, and R APIs.

# COMMAND ----------

# MAGIC %md
# MAGIC **Where MLflow runs are logged**
# MAGIC
# MAGIC All MLflow runs are logged to the active experiment, which can be set using any of the following ways:
# MAGIC
# MAGIC > - Use the [mlflow.set_experiment() command](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_experiment).
# MAGIC
# MAGIC > - Use the experiment_id parameter in the [mlflow.start_run() command](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run).
# MAGIC
# MAGIC > - Set one of the MLflow environment variables [MLFLOW_EXPERIMENT_NAME or MLFLOW_EXPERIMENT_ID](https://mlflow.org/docs/latest/cli.html#cmdoption-mlflow-run-arg-uri).
# MAGIC
# MAGIC If no active experiment is set, runs are logged to the notebook experiment.
# MAGIC
# MAGIC To log your experiment results to a remotely hosted MLflow Tracking server in a workspace other than the one in which you are running your experiment, set the tracking URI to reference the remote workspace with mlflow.set_tracking_uri(), and set the path to your experiment in the remote workspace by using mlflow.set_experiment().
# MAGIC
# MAGIC **Python**:
# MAGIC
# MAGIC *mlflow.set_tracking_uri(<uri_of_remote_workspace>)*
# MAGIC
# MAGIC *mlflow.set_experiment("path to experiment in remote workspace")*

# COMMAND ----------

# MAGIC %md
# MAGIC **Organize training runs with MLflow experiments**
# MAGIC
# MAGIC Experiments are units of organization for your model training runs. There are two types of experiments: workspace and notebook.
# MAGIC
# MAGIC > - You can create a workspace experiment from the Databricks Machine Learning UI or the MLflow API. Workspace experiments are not associated with any notebook, and any notebook can log a run to these experiments by using the experiment ID or the experiment name.
# MAGIC
# MAGIC > - A notebook experiment is associated with a specific notebook. Databricks automatically creates a notebook experiment if there is no active experiment when you start a run using mlflow.start_run().
# MAGIC
# MAGIC To see all of the experiments in a workspace that you have access to, click ![image!](https://docs.databricks.com/_images/experiments-icon.png) Experiments Icon Experiments in the sidebar. This icon appears only when you are in the machine learning persona.

# COMMAND ----------

# MAGIC %md
# MAGIC **Logging example notebook**
# MAGIC
# MAGIC This [notebook](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809225788/command/3558079809225790) (/Koantek_Demo_For_Challice/MLFlow/LogRunsToExperiment) shows how to log runs to a notebook experiment and to a workspace experiment. Only MLflow runs initiated within a notebook can be logged to the notebook experiment. MLflow runs launched from any notebook or from the APIs can be logged to a workspace experiment. For information about viewing logged runs, see View notebook experiment and View workspace experiment.

# COMMAND ----------

# MAGIC %md
# MAGIC **Create workspace experiment**
# MAGIC
# MAGIC This section describes how to create a workspace experiment using the Databricks UI. You can create a workspace experiment directly from the workspace or from the Experiments page.
# MAGIC
# MAGIC For instructions on logging runs to workspace experiments, see Logging example notebook.
# MAGIC
# MAGIC > 1. Click ![Workspace](https://docs.databricks.com/_images/workspace-icon.png) Icon Workspace in the sidebar.
# MAGIC
# MAGIC > 2. Go to the folder in which you want to create the experiment.
# MAGIC
# MAGIC > 3. Do one of the following:
# MAGIC
# MAGIC >> - Next to any folder, click Menu Dropdown on the right side of the text and select **Create > MLflow Experiment**.
# MAGIC
# MAGIC >> ![Create experiment](https://docs.databricks.com/_images/mlflow-experiments-create-aws.png)
# MAGIC
# MAGIC >> - In the workspace or a user folder, click Down Caret and select Create > MLflow Experiment.
# MAGIC
# MAGIC > 4. In the Create MLflow Experiment dialog, enter a name for the experiment and an optional artifact location. If you do not specify an artifact location, artifacts are stored in dbfs:/databricks/mlflow-tracking/<experiment-id>.
# MAGIC
# MAGIC >> Databricks supports DBFS, S3, and Azure Blob storage artifact locations.
# MAGIC
# MAGIC >> To store artifacts in S3, specify a URI of the form s3://<bucket>/<path>. MLflow obtains credentials to access S3 from your clusters’s instance profile. Artifacts stored in S3 do not appear in the MLflow UI; you must download them using an object storage client.
# MAGIC   
# MAGIC >> To store artifacts in Azure Blob storage, specify a URI of the form wasbs://<container>@<storage-account>.blob.core.windows.net/<path>. Artifacts stored in Azure Blob storage do not appear in the MLflow UI; you must download them using a blob storage client.
# MAGIC   
# MAGIC > 5. Click Create. An empty experiment appears.
# MAGIC
# MAGIC >> You can also create a new workspace experiment from the Experiments page. To create a new experiment, use the ![create experiment](https://docs.databricks.com/_images/create-expt-dropdown.png) drop-down drop-down menu. From the drop-down menu, you can select either an AutoML experiment or a blank (empty) experiment.
# MAGIC
# MAGIC >> - AutoML experiment. The Configure AutoML experiment page appears. For information about using AutoML, see Train ML models with the Databricks AutoML UI.
# MAGIC
# MAGIC >> - Blank experiment. The Create MLflow Experiment dialog appears. Enter a name and optional artifact location in the dialog to create a new workspace experiment. The default artifact location is dbfs:/databricks/mlflow-tracking/<experiment-id>.
# MAGIC
# MAGIC >> To log runs to this experiment, call mlflow.set_experiment() with the experiment path. The experiment path appears at the top of the experiment page. See Logging example notebook for details and an example notebook.
# MAGIC   
# MAGIC If you delete a notebook experiment using the API (for example, MlflowClient.tracking.delete_experiment() in Python), the notebook itself is moved into the Trash folder.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Demo Notebooks
# MAGIC
# MAGIC > - [MLflow quickstart part 1: training and logging](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809225638/command/3558079809225639) /Koantek_Demo_For_Challice/MLFlow/MLflow quickstart part 1: training and logging
# MAGIC > - [MLflowLoggingAPIPythonQuickstart](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809225882/command/3558079809225883) /Koantek_Demo_For_Challice/MLFlow/MLflowLoggingAPIPythonQuickstart

# COMMAND ----------

# MAGIC %md
# MAGIC **View experiment**
# MAGIC
# MAGIC Each experiment that you have access to appears on the experiments page. From this page, you can view any experiment. Click on an experiment name to display the experiment page.
# MAGIC
# MAGIC Additional ways to access the experiment page:
# MAGIC
# MAGIC > - You can access the experiment page for a workspace experiment from the workspace menu.
# MAGIC
# MAGIC > - You can access the experiment page for a notebook experiment from the notebook.
# MAGIC
# MAGIC **View workspace experiment**
# MAGIC
# MAGIC > 1. Click ![Workspace Icon](https://docs.databricks.com/_images/workspace-icon.png) Workspace in the sidebar.
# MAGIC
# MAGIC > 2. Go to the folder containing the experiment.
# MAGIC
# MAGIC > 3. Click the experiment name.
# MAGIC
# MAGIC **View notebook experiment**
# MAGIC
# MAGIC In the notebook’s right sidebar, click the Experiment icon ![Experiment icon](https://docs.databricks.com/_images/experiment.png).
# MAGIC
# MAGIC The Experiment Runs sidebar appears and shows a summary of each run associated with the notebook experiment, including run parameters and metrics. At the top of the sidebar is the name of the experiment that the notebook most recently logged runs to (either a notebook experiment or a workspace experiment).
# MAGIC
# MAGIC ![View run parameters and metrics](https://docs.databricks.com/_images/mlflow-notebook-revision.png)
# MAGIC
# MAGIC From the sidebar, you can navigate to the experiment page or directly to a run.
# MAGIC
# MAGIC > - To view the experiment, click External Link at the far right, next to Experiment Runs.
# MAGIC
# MAGIC > - To display a run, click the name of the run.
# MAGIC
# MAGIC **Manage experiments**
# MAGIC
# MAGIC You can rename, delete, or manage permissions for an experiment you own from the experiments page, the experiment page, or the workspace menu.
# MAGIC
# MAGIC **Change permissions for experiment**
# MAGIC
# MAGIC To change permissions for an experiment from the experiment page, click Share.
# MAGIC
# MAGIC ![Experiment page permissions button](https://docs.databricks.com/_images/expt-permission.png)
# MAGIC
# MAGIC You can change permissions for an experiment that you own from the experiments page. Click ![three button icon](https://docs.databricks.com/_images/three-button-icon.png) in the Actions column and select Permission.

# COMMAND ----------

# MAGIC %md
# MAGIC **Manage training code with MLflow runs**
# MAGIC
# MAGIC All MLflow runs are logged to the active experiment. If you have not explicitly set an experiment as the active experiment, runs are logged to the notebook experiment.
# MAGIC
# MAGIC **View runs**
# MAGIC
# MAGIC You can access a run either from its parent experiment page or directly from the notebook that created the run.
# MAGIC
# MAGIC From the experiment page, in the runs table, click the start time of a run.
# MAGIC
# MAGIC From the notebook, click ![External Link](https://docs.databricks.com/_images/external-link.png) next to the date and time of the run in the Experiment Runs sidebar.
# MAGIC
# MAGIC The run screen shows the parameters used for the run, the metrics resulting from the run, and any tags or notes. To display Notes, Parameters, Metrics, or Tags for this run, click right-pointing arrow to the left of the label.
# MAGIC
# MAGIC You also access artifacts saved from a run in this screen.
# MAGIC
# MAGIC ![View run](https://docs.databricks.com/_images/quick-start-nb-run.png)
# MAGIC
# MAGIC **Code snippets for prediction**
# MAGIC
# MAGIC If you log a model from a run, the model appears in the Artifacts section of this page. To display code snippets illustrating how to load and use the model to make predictions on Spark and pandas DataFrames, click the model name.
# MAGIC
# MAGIC ![predict code snippets](https://docs.databricks.com/_images/model-snippets.png)
# MAGIC
# MAGIC **View the notebook or Git project used for a run**
# MAGIC
# MAGIC To view the version of the notebook that created a run:
# MAGIC
# MAGIC > - On the experiment page, click the link in the Source column.
# MAGIC
# MAGIC > - On the run page, click the link next to Source.
# MAGIC
# MAGIC > - From the notebook, in the Experiment Runs sidebar, click the Notebook icon Notebook Version Icon in the box for that Experiment Run.
# MAGIC
# MAGIC The version of the notebook associated with the run appears in the main window with a highlight bar showing the date and time of the run.
# MAGIC
# MAGIC If the run was launched remotely from a Git project, click the link in the Git Commit field to open the specific version of the project used in the run. The link in the Source field opens the main branch of the Git project used in the run.
# MAGIC
# MAGIC **Add a tag to a run**
# MAGIC
# MAGIC Tags are key-value pairs that you can create and use later to search for runs.
# MAGIC
# MAGIC > 1. From the run page, click Tag icon if it is not already open. The tags table appears.
# MAGIC
# MAGIC ![tag table](https://docs.databricks.com/_images/tags-open.png)
# MAGIC
# MAGIC > 2. Click in the Name and Value fields and type the key and value for your tag.
# MAGIC
# MAGIC > 3. Click Add.
# MAGIC
# MAGIC ![add tag](https://docs.databricks.com/_images/tag-add.png)
# MAGIC
# MAGIC **Edit or delete a tag for a run**
# MAGIC
# MAGIC To edit or delete an existing tag, use the icons in the Actions column.
# MAGIC
# MAGIC ![tag actions](https://docs.databricks.com/_images/tag-edit-or-delete.png)
# MAGIC
# MAGIC **Reproduce the software environment of a run**
# MAGIC
# MAGIC You can reproduce the exact software environment for the run by clicking Reproduce Run. The following dialog appears:
# MAGIC
# MAGIC ![Reproduce run dialog](https://docs.databricks.com/_images/reproduce-run.png)
# MAGIC
# MAGIC With the default settings, when you click Confirm:
# MAGIC
# MAGIC > - The notebook is cloned to the location shown in the dialog.
# MAGIC
# MAGIC > - If the original cluster still exists, the cloned notebook is attached to the original cluster and the cluster is started.
# MAGIC
# MAGIC > - If the original cluster no longer exists, a new cluster with the same configuration, including any installed libraries, is created and started. The notebook is attached to the new cluster.
# MAGIC
# MAGIC You can select a different location for the cloned notebook and inspect the cluster configuration and installed libraries:
# MAGIC
# MAGIC > - To select a different folder to save the cloned notebook, click Edit Folder.
# MAGIC
# MAGIC > - To see the cluster spec, click View Spec. To clone only the notebook and not the cluster, uncheck this option.
# MAGIC
# MAGIC > - To see the libraries installed on the original cluster, click View Libraries. If you don’t care about installing the same libraries as on the original cluster, uncheck this option.
# MAGIC
# MAGIC **Manage runs**
# MAGIC
# MAGIC **Rename run**
# MAGIC
# MAGIC To rename a run, click ![three button icon](https://docs.databricks.com/_images/three-button-icon.png) at the upper right corner of the run page and select Rename.
# MAGIC
# MAGIC **Filter runs**
# MAGIC
# MAGIC You can search for runs based on parameter or metric values. You can also search for runs by tag.
# MAGIC
# MAGIC > - To search for runs that match an expression containing parameter and metric values, enter a query in the search field and click Search. Some query syntax examples are:
# MAGIC
# MAGIC > metrics.r2 > 0.3
# MAGIC
# MAGIC > params.elasticNetParam = 0.5
# MAGIC
# MAGIC > params.elasticNetParam = 0.5 AND metrics.avg_areaUnderROC > 0.3
# MAGIC
# MAGIC > - To search for runs by tag, enter tags in the format: tags.<key>="<value>". String values must be enclosed in quotes as shown.
# MAGIC
# MAGIC > tags.estimator_name="RandomForestRegressor"
# MAGIC
# MAGIC > tags.color="blue" AND tags.size=5
# MAGIC
# MAGIC Both keys and values can contain spaces. If the key includes spaces, you must enclose it in backticks as shown.
# MAGIC
# MAGIC tags.'my custom tag' = "my value"
# MAGIC   
# MAGIC You can also filter runs based on their state (Active or Deleted) and based on whether a model version is associated with the run. To do this, click Filter to the right of the Search box. The State and Linked Models drop-down menus appear. Make your selections from the drop-down menus.
# MAGIC
# MAGIC ![Filter runs](https://docs.databricks.com/_images/quick-start-nb-experiment.png)
# MAGIC   
# MAGIC **Compare runs**
# MAGIC You can compare runs from a single experiment or from multiple experiments. The Comparing Runs page presents information about the selected runs in graphic and tabular formats.
# MAGIC
# MAGIC **Compare runs from a single experiment**
# MAGIC   
# MAGIC > 1. On the experiment page, select two or more runs by clicking in the checkbox to the left of the run, or select all runs by checking the box at the top of the column.
# MAGIC
# MAGIC > 2. Click Compare. The Comparing `<N>` Runs screen appears.
# MAGIC
# MAGIC Compare runs from multiple experiments
# MAGIC   
# MAGIC > 1. On the experiments page, select the experiments you want to compare by clicking in the box at the left of the experiment name.
# MAGIC
# MAGIC > 2. Click Compare (n) (n is the number of experiments you selected). A screen appears showing all of the runs from the experiments you selected.
# MAGIC
# MAGIC > 3. Select two or more runs by clicking in the checkbox to the left of the run, or select all runs by checking the box at the top of the column.
# MAGIC
# MAGIC > 4. Click Compare. The Comparing `<N>` Runs screen appears.
# MAGIC
# MAGIC Use the Comparing Runs page
# MAGIC   
# MAGIC The Comparing Runs page shows visualizations of run results and tables of run information, run parameters, and metrics.
# MAGIC
# MAGIC To create a visualization:
# MAGIC
# MAGIC > 1. Select the plot type (Parallel Coordinates Plot, Scatter Plot, or Contour Plot).
# MAGIC
# MAGIC > 2. For a Parallel Coordinates Plot, select the parameters and metrics to plot. For a Scatter Plot or Contour Plot, select the parameter or metric to display on each axis.
# MAGIC
# MAGIC ![compare runs page visualization](https://docs.databricks.com/_images/mlflow-run-comparison-viz.png)
# MAGIC   
# MAGIC The Parameters and Metrics tables display the run parameters and metrics from all selected runs. The columns in these tables are identified by the Run details table immediately above. For simplicity, you can hide parameters and metrics that are identical in all selected runs by toggling Show diff only button.
# MAGIC
# MAGIC ![compare runs page tables](https://docs.databricks.com/_images/mlflow-run-comparison-table.png)
# MAGIC
# MAGIC **Download runs**
# MAGIC   
# MAGIC > 1. Select one or more runs.
# MAGIC
# MAGIC > 2. Click Download CSV. A CSV file containing the following fields downloads:
# MAGIC
# MAGIC
# MAGIC Run ID,Name,Source Type,Source Name,User,Status,<parameter1>,<parameter2>,...,<metric1>,<metric2>,...
# MAGIC   
# MAGIC   
# MAGIC **Delete runs**
# MAGIC   
# MAGIC > 1. In the experiment, select one or more runs by clicking in the checkbox to the left of the run.
# MAGIC
# MAGIC > 2. Click Delete.
# MAGIC
# MAGIC > 3. If the run is a parent run, decide whether you also want to delete descendant runs. This option is selected by default.
# MAGIC
# MAGIC > 4. Click Delete to confirm or Cancel to cancel. Deleted runs are saved for 30 days. To display deleted runs, select Deleted in the State field.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze MLflow runs programmatically
# MAGIC
# MAGIC You can access MLflow run data programmatically using the following two DataFrame APIs:
# MAGIC
# MAGIC > - The MLflow Python client [search_runs API](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs) returns a pandas DataFrame.
# MAGIC
# MAGIC > - The [MLflow experiment](https://docs.databricks.com/external-data/mlflow-experiment.html#mlflow-exp-datasource) data source returns an Apache Spark DataFrame.
# MAGIC
# MAGIC This example demonstrates how to use the MLflow Python client to build a dashboard that visualizes changes in evaluation metrics over time, tracks the number of runs started by a specific user, and measures the total number of runs across all users:
# MAGIC
# MAGIC [Build dashboards with the MLflow Search API](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809225845/command/3558079809225846) /Koantek_Demo_For_Challice/MLFlow/MLflow Search API Dashboards

# COMMAND ----------

# MAGIC %md
# MAGIC **MLflow tracking examples**
# MAGIC
# MAGIC The following notebooks demonstrate how to train several types of models and track the training data in MLflow and how to store tracking data in Delta Lake.
# MAGIC
# MAGIC > - [Track scikit-learn model training with MLflow](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809225638) /Koantek_Demo_For_Challice/MLFlow/MLflow quickstart part 1: training and logging
# MAGIC > - [Train a PyTorch model](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809225867/command/3558079809225868) /Koantek_Demo_For_Challice/MLFlow/MLflow: Train with PyTorch
# MAGIC > - [Train a PySpark model and save in MLeap format](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809225831/command/3558079809225832) /Koantek_Demo_For_Challice/MLFlow/MLflow: Train PySpark Model and Log in MLeap Format
# MAGIC > - [Track ML Model training data with Delta Lake](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809225801/command/3558079809225802) /Koantek_Demo_For_Challice/MLFlow/Tracking ML Model Training with MLflow and Delta Lake

# COMMAND ----------

# MAGIC %md
# MAGIC **Databricks Autologging**
# MAGIC
# MAGIC Databricks Autologging is a no-code solution that extends [MLflow automatic logging](https://mlflow.org/docs/latest/tracking.html#automatic-logging) to deliver automatic experiment tracking for machine learning training sessions on Databricks. With Databricks Autologging, model parameters, metrics, files, and lineage information are automatically captured when you train models from a variety of popular machine learning libraries. Training sessions are recorded as [MLflow tracking runs](https://docs.databricks.com/mlflow/tracking.html). Model files are also tracked so you can easily log them to the [MLflow Model Registry](https://docs.databricks.com/mlflow/model-registry.html) and deploy them for real-time scoring with [Serverless Real-Time Inference](https://docs.databricks.com/machine-learning/model-inference/serverless/serverless-real-time-inference.html).
# MAGIC
# MAGIC The following video shows Databricks Autologging with a scikit-learn model training session in an interactive Python notebook. Tracking information is automatically captured and displayed in the Experiment Runs sidebar and in the MLflow UI.
# MAGIC
# MAGIC ![Autologging example](https://docs.databricks.com/_images/autologging-example.gif)
# MAGIC
# MAGIC
# MAGIC
# MAGIC **How it works**
# MAGIC
# MAGIC When you attach an interactive Python notebook to a Databricks cluster, Databricks Autologging calls mlflow.autolog() to set up tracking for your model training sessions. When you train models in the notebook, model training information is automatically tracked with MLflow Tracking. For information about how this model training information is secured and managed, see Security and data management.
# MAGIC
# MAGIC The default configuration for the mlflow.autolog() call is:
# MAGIC
# MAGIC Python Code:
# MAGIC
# MAGIC `mlflow.autolog(`
# MAGIC
# MAGIC `    log_input_examples=False,`
# MAGIC     
# MAGIC `    log_model_signatures=True,`
# MAGIC     
# MAGIC `    log_models=True,`
# MAGIC     
# MAGIC `    disable=False,`
# MAGIC     
# MAGIC `    exclusive=True,`
# MAGIC     
# MAGIC `    disable_for_unsupported_versions=True,`
# MAGIC     
# MAGIC `    silent=True`
# MAGIC     
# MAGIC `)`
# MAGIC
# MAGIC **Customize logging behavior**
# MAGIC
# MAGIC To customize logging, use mlflow.autolog(). This function provides configuration parameters to enable model logging (log_models), collect input examples (log_input_examples), configure warnings (silent), and more.
# MAGIC
# MAGIC **Track additional content**
# MAGIC
# MAGIC To track additional metrics, parameters, files, and metadata with MLflow runs created by Databricks Autologging, follow these steps in a Databricks interactive Python notebook:
# MAGIC
# MAGIC > 1. Call mlflow.autolog() with exclusive=False.
# MAGIC
# MAGIC > 2. Start an MLflow run using mlflow.start_run(). You can wrap this call in with mlflow.start_run(); when you do this, the run is ended automatically after it completes.
# MAGIC
# MAGIC > 3. Use MLflow Tracking methods, such as mlflow.log_param(), to track pre-training content.
# MAGIC
# MAGIC > 4. Train one or more machine learning models in a framework supported by Databricks Autologging.
# MAGIC
# MAGIC > 5. Use MLflow Tracking methods, such as mlflow.log_metric(), to track post-training content.
# MAGIC
# MAGIC > 6. If you did not use with mlflow.start_run() in Step 2, end the MLflow run using mlflow.end_run().
# MAGIC
# MAGIC For example:
# MAGIC
# MAGIC Python Code:
# MAGIC
# MAGIC `import mlflow`
# MAGIC
# MAGIC `mlflow.autolog(exclusive=False)`
# MAGIC
# MAGIC `with mlflow.start_run():`
# MAGIC
# MAGIC `  mlflow.log_param("example_param", "example_value")`
# MAGIC
# MAGIC `  # <your model training code here>`
# MAGIC
# MAGIC `  mlflow.log_param("example_metric", 5)`
# MAGIC   
# MAGIC
# MAGIC
# MAGIC **Disable Databricks Autologging**
# MAGIC
# MAGIC To disable Databricks Autologging in a Databricks interactive Python notebook, call mlflow.autolog() with disable=True:
# MAGIC
# MAGIC Python Code:
# MAGIC
# MAGIC
# MAGIC `import mlflow`
# MAGIC
# MAGIC `mlflow.autolog(disable=True)
# MAGIC `
# MAGIC
# MAGIC Administrators can also disable Databricks Autologging for all clusters in a workspace from the Advanced tab of the [admin console](https://docs.databricks.com/administration-guide/admin-console.html). Clusters must be restarted for this change to take effect.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log, load, register, and deploy MLflow models
# MAGIC
# MAGIC An MLflow Model is a standard format for packaging machine learning models that can be used in a variety of downstream tools—for example, batch inference on Apache Spark or real-time serving through a REST API. The format defines a convention that lets you save a model in different flavors (python-function, pytorch, sklearn, and so on), that can be understood by different model serving and inference platforms.
# MAGIC
# MAGIC **Log and load models**
# MAGIC
# MAGIC With Databricks Runtime 8.4 ML and above, when you log a model, MLflow automatically logs requirements.txt and conda.yaml files. You can use these files to recreate the model development environment and reinstall dependencies using virtualenv (recommended) or conda.
# MAGIC
# MAGIC Important
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
# MAGIC For more information on the log_model() API, see the MLflow documentation for the model flavor you are working with, for example, log_model for scikit-learn.
# MAGIC
# MAGIC For more information on conda.yaml files, see the MLflow documentation.
# MAGIC
# MAGIC **API commands**
# MAGIC
# MAGIC To log a model to the MLflow [tracking server](https://docs.databricks.com/mlflow/tracking.html), use mlflow.<model-type>.log_model(model, ...).
# MAGIC
# MAGIC To load a previously logged model for inference or further development, use mlflow.<model-type>.load_model(modelpath), where modelpath is one of the following:
# MAGIC
# MAGIC > - a run-relative path (such as runs:/{run_id}/{model-path})
# MAGIC
# MAGIC > - a DBFS path
# MAGIC
# MAGIC > - a [registered model](https://docs.databricks.com/mlflow/model-registry.html) path (such as models:/{model_name}/{model_stage}).
# MAGIC
# MAGIC For a complete list of options for loading MLflow models, see [Referencing Artifacts in the MLflow documentation](https://www.mlflow.org/docs/latest/concepts.html#artifact-locations).
# MAGIC
# MAGIC For Python MLflow models, an additional option is to use mlflow.pyfunc.load_model() to load the model as a generic Python function. You can use the following code snippet to load the model and score data points.
# MAGIC
# MAGIC
# MAGIC Python Code:
# MAGIC   
# MAGIC   `model = mlflow.pyfunc.load_model(model_path)`
# MAGIC   
# MAGIC   `model.predict(model_input)`
# MAGIC   
# MAGIC As an alternative, you can export the model as an Apache Spark UDF to use for scoring on a Spark cluster, either as a batch job or as a real-time Spark Streaming job.
# MAGIC   
# MAGIC
# MAGIC Python Code:
# MAGIC   
# MAGIC   `# load input data table as a Spark DataFrame`
# MAGIC   
# MAGIC   `input_data = spark.table(input_table_name)`
# MAGIC   
# MAGIC   `model_udf = mlflow.pyfunc.spark_udf(model_path)`
# MAGIC   
# MAGIC   `df = input_data.withColumn("prediction", model_udf())`
# MAGIC   
# MAGIC **Log model dependencies**
# MAGIC   
# MAGIC To accurately load a model, you should make sure the model dependencies are loaded with the correct versions into the notebook environment. In Databricks Runtime 10.5 ML and above, MLflow warns you if a mismatch is detected between the current environment and the model’s dependencies.
# MAGIC
# MAGIC Additional functionality to simplify restoring model dependencies is included in Databricks Runtime 11.0 ML and above. In Databricks Runtime 11.0 ML and above, for pyfunc flavor models, you can call mlflow.pyfunc.get_model_dependencies to retrieve and download the model dependencies. This function returns a path to the dependencies file which you can then install by using %pip install <file-path>. When you load a model as a PySpark UDF, specify env_manager="virtualenv" in the mlflow.pyfunc.spark_udf call. This restores model dependencies in the context of the PySpark UDF and does not affect the outside environment.
# MAGIC
# MAGIC You can also use this functionality in Databricks Runtime 10.5 or below by manually installing MLflow version 1.25.0 or above:
# MAGIC
# MAGIC Python Code:
# MAGIC   
# MAGIC   `%pip install "mlflow>=1.25.0"`
# MAGIC   
# MAGIC For additional information on how to log model dependencies (Python and non-Python) and artifacts, see [Log model dependencies](https://docs.databricks.com/mlflow/log-model-dependencies.html).
# MAGIC
# MAGIC Learn how to log model dependencies and custom artifacts for model serving:
# MAGIC
# MAGIC > - [Deploy models with dependencies](https://docs.databricks.com/mlflow/log-model-dependencies.html#deploy-dependencies)
# MAGIC
# MAGIC > - [Use custom Python libraries with Serverless Real-Time Inference](https://docs.databricks.com/machine-learning/model-inference/serverless/private-libraries-serverless-model-serving.html)
# MAGIC
# MAGIC > - [Package custom artifacts for Serverless Real-Time Inference](https://docs.databricks.com/machine-learning/model-inference/serverless/serverless-model-serving-custom-artifacts.html)
# MAGIC
# MAGIC > - [Log model dependencies](https://docs.databricks.com/mlflow/log-model-dependencies.html)
# MAGIC   
# MAGIC **Automatically generated code snippets in the MLflow UI**
# MAGIC   
# MAGIC When you log a model in a Databricks notebook, Databricks automatically generates code snippets that you can copy and use to load and run the model. To view these code snippets:
# MAGIC
# MAGIC > 1. Navigate to the Runs screen for the run that generated the model. (See View notebook experiment for how to display the Runs screen.)
# MAGIC
# MAGIC > 2. Scroll to the Artifacts section.
# MAGIC
# MAGIC > 3. Click the name of the logged model. A panel opens to the right showing code you can use to load the logged model and make predictions on Spark or pandas DataFrames.
# MAGIC
# MAGIC ![Artifact panel code snippets](https://docs.databricks.com/_images/code-snippets.png)
# MAGIC   
# MAGIC **Examples**
# MAGIC   
# MAGIC For examples of logging models, see the examples in Track machine learning training runs examples. For an example of loading a logged model for inference, see the following example.
# MAGIC
# MAGIC [Model inference example](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809226004/command/3558079809226005) /Koantek_Demo_For_Challice/MLFlow/MLflow quickstart part 2: inference
# MAGIC   
# MAGIC **Register models in the Model Registry**
# MAGIC   
# MAGIC You can register models in the MLflow Model Registry, a centralized model store that provides a UI and set of APIs to manage the full lifecycle of MLflow Models. For general information about the Model Registry, see [MLflow Model Registry on Databricks](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809226039/command/3558079809226040) /Koantek_Demo_For_Challice/MLFlow/MLflow_Model_Registry_on_Databricks. For instructions on how to use the Model Registry to manage models in Databricks, see [Manage model lifecycle](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809226041/command/3558079809226042) /Koantek_Demo_For_Challice/MLFlow/Manage_model_lifecycle.
# MAGIC
# MAGIC To register a model using the API, use 
# MAGIC   
# MAGIC   mlflow.register_model("runs:/{run_id}/{model-path}", "{registered-model-name}").
# MAGIC
# MAGIC **Save models to DBFS**
# MAGIC   
# MAGIC To save a model locally, use mlflow.<model-type>.save_model(model, modelpath). modelpath must be a DBFS path. For example, if you use a DBFS location dbfs:/my_project_models to store your project work, you must use the model path /dbfs/my_project_models:
# MAGIC
# MAGIC Python Code:
# MAGIC   
# MAGIC   `modelpath = "/dbfs/my_project_models/model-%f-%f" % (alpha, l1_ratio)`
# MAGIC   
# MAGIC   `mlflow.sklearn.save_model(lr, modelpath)`
# MAGIC   
# MAGIC For MLlib models, use [ML Pipelines](https://spark.apache.org/docs/latest/ml-pipeline.html#ml-persistence-saving-and-loading-pipelines).
# MAGIC
# MAGIC **Download model artifacts**
# MAGIC   
# MAGIC You can download the logged model artifacts (such as model files, plots, and metrics) for a registered model with various APIs.
# MAGIC
# MAGIC [Python API](https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.get_model_version_download_uri) example:
# MAGIC
# MAGIC Python Code:
# MAGIC   
# MAGIC `from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository`
# MAGIC   
# MAGIC
# MAGIC 'model_uri = MlflowClient.get_model_version_download_uri(model_name, model_version)`
# MAGIC   
# MAGIC `ModelsArtifactRepository(model_uri).download_artifacts(artifact_path="")`
# MAGIC   
# MAGIC   
# MAGIC [Java API](https://mlflow.org/docs/latest/java_api/org/mlflow/tracking/MlflowClient.html#downloadModelVersion-java.lang.String-java.lang.String-) example:
# MAGIC
# MAGIC Java Code:
# MAGIC   
# MAGIC `MlflowClient mlflowClient = new MlflowClient();`
# MAGIC   
# MAGIC `// Get the model URI for a registered model version.`
# MAGIC   
# MAGIC `String modelURI = mlflowClient.getModelVersionDownloadUri(modelName, modelVersion);`
# MAGIC   
# MAGIC
# MAGIC `// Or download the model artifacts directly.`
# MAGIC   
# MAGIC `File modelFile = mlflowClient.downloadModelVersion(modelName, modelVersion);`
# MAGIC   
# MAGIC   
# MAGIC [CLI command](https://www.mlflow.org/docs/latest/cli.html#mlflow-artifacts-download) example:
# MAGIC
# MAGIC `mlflow artifacts download --artifact-uri models:/<name>/<version|stage>`
# MAGIC   
# MAGIC   
# MAGIC **Deploy models for online serving**
# MAGIC   
# MAGIC You can use [Serverless Real-Time Inference](https://docs.databricks.com/machine-learning/model-inference/serverless/serverless-real-time-inference.html) to host machine learning models from the Model Registry as REST endpoints. These endpoints are updated automatically based on the availability of model versions and their stages.
# MAGIC
# MAGIC You can also deploy a model to third-party serving frameworks using [MLflow’s built-in deployment tools](https://mlflow.org/docs/latest/models.html#built-in-deployment-tools).
# MAGIC
# MAGIC See the following examples:
# MAGIC
# MAGIC > - [scikit-learn model deployment on SageMaker](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809226165/command/3558079809226169) /Koantek_Demo_For_Challice/MLFlow/MLflow quickstart part 2: serving models using Amazon SageMaker
# MAGIC > - [MLeap model deployment on SageMaker](https://dbc-00646cc6-55ee.cloud.databricks.com/?o=512145377436569#notebook/3558079809226206/command/3558079809226207) /Koantek_Demo_For_Challice/MLFlow/MLflow: Deployment with MLeap 