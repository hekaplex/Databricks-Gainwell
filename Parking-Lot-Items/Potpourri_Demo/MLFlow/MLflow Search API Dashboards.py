# Databricks notebook source
# MAGIC %md ##Building Dashboards with the MLflow Search API
# MAGIC
# MAGIC This notebook demonstrates how to use the [mlflow.search_runs](https://www.mlflow.org/docs/latest/search-syntax.html#programmatically-searching-runs) API to generate custom dashboards.
# MAGIC
# MAGIC The notebook contains the following sections:
# MAGIC
# MAGIC #### Setup
# MAGIC * Launch a Python 3 cluster running Databricks Runtime ML
# MAGIC
# MAGIC #### Get existing experiment
# MAGIC * Generate dashboards from existing experiments.
# MAGIC
# MAGIC #### Dashboards
# MAGIC * Visualize changes in evaluation metrics over time
# MAGIC   * The example utilizes Mean Absolute Error (MAE) to illustrate how to visualize changes in metrics over time. If you choose to run this notebook on your own experiment, be sure your experiment calculates and logs the MAE metric prior to trying out this example. 
# MAGIC * Track the number of runs kicked off by a particular user
# MAGIC * Measure the total number of runs across all users

# COMMAND ----------

# MAGIC %md ## Setup

# COMMAND ----------

# MAGIC %md 
# MAGIC 1. Ensure you are using or create a cluster specifying 
# MAGIC   * **Databricks Runtime Version:** Any version of Databricks Runtime ML (e.g. Databricks Runtime 6.3 ML)
# MAGIC   * **Python Version:** Python 3
# MAGIC 1. Attach this notebook to the cluster.

# COMMAND ----------

# MAGIC %md ## Get existing experiment ID

# COMMAND ----------

# MAGIC %md To plot summaries of existing experiment data, you need the notebook path in which your experiment was run. 
# MAGIC 1. Find and select the experiment of interest via the Experiments page in the left sidebar
# MAGIC 1. On the experiment page, select the copy icon to the right of the experiment path at the top of the page.
# MAGIC 1. Paste this path into the following code cell as your experiment_name variable-- replace the text experiment_notebook_path, but keep the single quotations.  
# MAGIC
# MAGIC The following shows how to get the experiment ID of an existing experiment in your workspace with the experiment notebook path and [get_experiment_by_name()](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.get_experiment_by_name).

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

# Use the experiment notebook path to get the experiment ID
experiment_name = 'experiment_notebook_path'
experiment_id= MlflowClient().get_experiment_by_name(experiment_name).experiment_id


# COMMAND ----------

# MAGIC %md ## Construct Graphs for the Dashboard

# COMMAND ----------

import pandas as pd
from datetime import datetime, timedelta

# COMMAND ----------

# MAGIC %md ### Visualize changes in evaluation metrics over time
# MAGIC
# MAGIC This example tracks progress towards optimizing Mean Absolute Error (MAE) over recent days of experiment runs.
# MAGIC
# MAGIC You may wish to modify `earliest_start_time`, which restricts which runs are displayed.

# COMMAND ----------


runs = mlflow.search_runs(experiment_ids=experiment_id)

earliest_start_time = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
recent_runs = runs[runs.start_time >= earliest_start_time]
pd.options.mode.chained_assignment = None  # Suppress pandas warning
recent_runs['Run Date'] = recent_runs.start_time.dt.floor(freq='D')
best_runs_per_day_idx = recent_runs.groupby(['Run Date'])['metrics.mae'].idxmin()
best_runs = recent_runs.loc[best_runs_per_day_idx]

# COMMAND ----------

# DBTITLE 1,Best Performing Run for the Past 2 Weeks
display(best_runs[['Run Date', 'metrics.mae']])

# COMMAND ----------

# MAGIC %md
# MAGIC You can add this chart to a dashboard by clicking the Dashboard icon at the top right of the cell and selecting **Add to New Dashboard**.

# COMMAND ----------

# MAGIC %md ### Track the number of runs started by a specific user
# MAGIC
# MAGIC This example calculates the number of runs started by a specific user each day.
# MAGIC
# MAGIC You may also wish to change:
# MAGIC
# MAGIC * `filter_string`: Defaults to selecting runs started by you
# MAGIC * `earliest_start_time`: Restricts which runs are displayed

# COMMAND ----------

dbutils.widgets.text("User", "")

# COMMAND ----------

user = dbutils.widgets.get("User")

if user == '':
  print('Enter a user in the widget at the top of this page (for example, "username@example.com"). Defaults to all users.')
  filter_string = None
else:
  filter_string=f'tags.mlflow.user = "{user}"'
  
runs = mlflow.search_runs(
  experiment_ids=experiment_id,
  filter_string=filter_string
)
earliest_start_time = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
# If there are no runs, the 'start_time' column will have type float64. Cast it to a datetime64
# before comparing with earliest_start_time.
runs['start_time'] = runs['start_time'].astype('datetime64[ns]')
recent_runs = runs[runs.start_time >= earliest_start_time]

recent_runs['Run Date'] = recent_runs.start_time.dt.floor(freq='D')

runs_per_day = recent_runs.groupby(['Run Date']).count()[['run_id']].reset_index()
runs_per_day['Run Date'] = runs_per_day['Run Date'].dt.strftime('%Y-%m-%d')
runs_per_day.rename({ 'run_id': 'Number of Runs' }, axis='columns', inplace=True)

# COMMAND ----------

# DBTITLE 1,Number of Recent Runs Started by a User
if runs_per_day.shape[0] > 0:
  display(runs_per_day)

# COMMAND ----------

# MAGIC %md ### Measure the total number of runs across all users
# MAGIC
# MAGIC This example calculates the total number of experiment runs created during each month of 2019.

# COMMAND ----------

runs = mlflow.search_runs(experiment_ids=experiment_id)

# COMMAND ----------

runs_2019 = runs[(runs.start_time < '2020-01-01') & (runs.start_time >= '2019-01-01')]
runs_2019['month'] = runs_2019.start_time.dt.month_name()
runs_2019['month_i'] = runs_2019.start_time.dt.month

runs_per_month = runs_2019.groupby(['month_i', 'month']).count()[['run_id']].reset_index('month')
runs_per_month.rename({ 'run_id': 'Number of Runs' }, axis='columns', inplace=True)

# COMMAND ----------

# DBTITLE 0,Total Number of Runs in 2019
if runs_per_month.shape[0] > 0:
  display(runs_per_month)