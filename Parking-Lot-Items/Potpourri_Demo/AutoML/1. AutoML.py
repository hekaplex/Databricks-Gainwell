# Databricks notebook source
# MAGIC %md
# MAGIC ### How Databricks AutoML works
# MAGIC
# MAGIC
# MAGIC This article details how Databricks AutoML works and its implementation of concepts like missing value imputation and large data sampling.
# MAGIC
# MAGIC Databricks AutoML performs the following:
# MAGIC
# MAGIC > 1. Prepares the dataset for model training. For example, AutoML carries out imbalanced data detection for classification problems prior to model training.
# MAGIC
# MAGIC > 2. Iterates to train and tune multiple models, where each model is constructed from open source components and can easily be edited and integrated into your machine learning pipelines.
# MAGIC
# MAGIC >> - AutoML automatically distributes hyperparameter tuning trials across the worker nodes of a cluster.
# MAGIC
# MAGIC >> - With Databricks Runtime 9.1 LTS ML or above, AutoML automatically samples your dataset if it is too large to fit into the memory of a single worker node.
# MAGIC
# MAGIC > 3. Evaluates models based on algorithms from the scikit-learn, xgboost, LightGBM, Prophet, and ARIMA packages.
# MAGIC
# MAGIC > 4. Displays the results and provides a Python notebook with the source code for each trial run so you can review, reproduce, and modify the code. AutoML also calculates summary statistics on your dataset and saves this information in a notebook that you can review later.

# COMMAND ----------

# MAGIC %md
# MAGIC ### AutoML algorithms
# MAGIC
# MAGIC Databricks AutoML trains and evaluates models based on the algorithms in the following table.
# MAGIC
# MAGIC Note
# MAGIC
# MAGIC For classification and regression models, the decision tree, random forests, logistic regression and linear regression with stochastic gradient descent algorithms are based on scikit-learn.
# MAGIC
# MAGIC ![Automl](https://dbc-00646cc6-55ee.cloud.databricks.com/files/automl.PNG?o=512145377436569)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Supported data feature types
# MAGIC
# MAGIC Feature types not listed below are not supported. For example, images are not supported.
# MAGIC
# MAGIC The following feature types are supported:
# MAGIC
# MAGIC Numeric (ByteType, ShortType, IntegerType, LongType, FloatType, and DoubleType)
# MAGIC
# MAGIC Boolean
# MAGIC
# MAGIC String (categorical or English text)
# MAGIC
# MAGIC Timestamps (TimestampType, DateType)
# MAGIC
# MAGIC ArrayType[Numeric] (Databricks Runtime 10.4 LTS ML and above)
# MAGIC
# MAGIC DecimalType (Databricks Runtime 11.3 LTS ML and above)
# MAGIC
# MAGIC ### Split data into train/validation/test sets
# MAGIC
# MAGIC With Databricks Runtime 10.1 ML and above, you can specify a time column to use for the training/validation/testing data split for classification and regression problems. If you specify this column, the dataset is split into training, validation, and test sets by time. The earliest points are used for training, the next earliest for validation, and the latest points are used as a test set.
# MAGIC
# MAGIC In Databricks Runtime 10.1 ML, the time column must be a timestamp or integer column. In Databricks Runtime 10.2 ML and above, you can also select a string column.
# MAGIC
# MAGIC ### Sampling large datasets
# MAGIC
# MAGIC Note
# MAGIC
# MAGIC Sampling is not applied to forecasting problems.
# MAGIC
# MAGIC Although AutoML distributes hyperparameter tuning trials across the worker nodes of a cluster, each model is trained on a single worker node.
# MAGIC
# MAGIC AutoML automatically estimates the memory required to load and train your dataset and samples the dataset if necessary.
# MAGIC
# MAGIC In Databricks Runtime 9.1 LTS ML through Databricks Runtime 10.5 ML, the sampling fraction does not depend on the cluster’s node type or the amount of memory on each node.
# MAGIC
# MAGIC In Databricks Runtime 11.x ML:
# MAGIC
# MAGIC The sampling fraction increases for worker nodes that have more memory per core. You can increase the sample size by choosing a memory optimized instance type.
# MAGIC
# MAGIC You can further increase the sample size by choosing a larger value for spark.task.cpus in the Spark configuration for the cluster. The default setting is 1; the maximum value is the number of CPUs on the worker node. When you increase this value, the sample size is larger, but fewer trials run in parallel. For example, in a machine with 4 cores and 64GB total RAM, the default spark.task.cpus=1 runs 4 trials per worker with each trial limited to 16GB RAM. If you set spark.task.cpus=4, each worker runs only one trial but that trial can use 64GB RAM.
# MAGIC
# MAGIC In Databricks Runtime 12.0 ML and above, AutoML can train on larger datasets by allocating more CPU cores per training task. You can increase the sample size by choosing an instance size with larger total memory.
# MAGIC
# MAGIC In Databricks Runtime 11.0 ML and above, if AutoML sampled the dataset, the sampling fraction is shown in the Overview tab in the UI.
# MAGIC
# MAGIC For classification problems, AutoML uses the PySpark sampleBy method for stratified sampling to preserve the target label distribution.
# MAGIC
# MAGIC For regression problems, AutoML uses the PySpark sample method.
# MAGIC
# MAGIC ### Imbalanced dataset support for classification problems
# MAGIC
# MAGIC In Databricks Runtime 11.2 ML and above, if AutoML detects that a dataset is imbalanced, it tries to reduce the imbalance of the training dataset by downsampling the major class(es) and adding class weights. AutoML only balances the training dataset and does not balance the test and validation datasets. Doing so ensures that the model performance is always evaluated on the non-enriched dataset with the true input class distribution.
# MAGIC
# MAGIC To balance an imbalanced training dataset, AutoML uses class weights that are inversely related to the degree by which a given class is downsampled. For example, if a training dataset with 100 samples has 95 samples belonging to class A and 5 samples belonging to class B, AutoML reduces this imbalance by downsampling class A to 70 samples, that is downsampling class A by a ratio of 70/95 or 0.736, while keeping the number of samples in class B at 5. To ensure that the final model is correctly calibrated and the probability distribution of the model output is the same as that of the input, AutoML scales up the class weight for class A by the ratio 1/0.736, or 1.358, while keeping the weight of class B as 1. AutoML then uses these class weights in model training as a parameter to ensure that the samples from each class are weighted appropriately when training the model.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Semantic type detection
# MAGIC >Note
# MAGIC >Semantic type detection is not applied to forecasting problems.<br>
# MAGIC >AutoML does not perform semantic type detection for columns that have custom imputation methods specified
# MAGIC
# MAGIC With Databricks Runtime 9.1 LTS ML and above, AutoML tries to detect whether columns have a semantic type that is different from the Spark or pandas data type in the table schema. AutoML treats these columns as the detected semantic type. These detections are best effort and might miss the existence of semantic types in some cases. You can also manually set the semantic type of a column or tell AutoML not to apply semantic type detection to a column using annotations.
# MAGIC
# MAGIC Specifically, AutoML makes these adjustments:
# MAGIC
# MAGIC - String and integer columns that represent date or timestamp data are treated as a timestamp type.
# MAGIC
# MAGIC - String columns that represent numeric data are treated as a numeric type.
# MAGIC
# MAGIC With Databricks Runtime 10.1 ML and above, AutoML also makes these adjustments:
# MAGIC
# MAGIC - Numeric columns that contain categorical IDs are treated as a categorical feature.
# MAGIC
# MAGIC - String columns that contain English text are treated as a text feature.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Semantic type annotations
# MAGIC With Databricks Runtime 10.1 ML and above, you can manually control the assigned semantic type by placing a semantic type annotation on a column. To manually annotate the semantic type of column <column_name> as <semantic_type>, use the following syntax:
# MAGIC
# MAGIC Python 
# MAGIC metadata_dict = df.schema["<column_name>"].metadata <br>
# MAGIC metadata_dict["spark.contentAnnotation.semanticType"] = "<semantic_type>" <br>
# MAGIC df = df.withMetadata("<column_name>", metadata_dict) <br>
# MAGIC
# MAGIC <semantic_type> can be one of the following:
# MAGIC
# MAGIC - categorical: The column contains categorical values (for example, numerical values that should be treated as IDs).
# MAGIC
# MAGIC - numeric: The column contains numeric values (for example, string values that can be parsed into numbers).
# MAGIC
# MAGIC - datetime: The column contains timestamp values (string, numerical, or date values that can be converted into timestamps).
# MAGIC
# MAGIC - text: The string column contains English text.
# MAGIC
# MAGIC To disable semantic type detection on a column, use the special keyword annotation native.
# MAGIC
# MAGIC ### Shapley values (SHAP) for model explainability
# MAGIC >Note
# MAGIC > For MLR 11.1 and below, SHAP plots are not generated, if the dataset contains a datetime column.
# MAGIC
# MAGIC The notebooks produced by AutoML regression and classification runs include code to calculate Shapley values. Shapley values are based in game theory and estimate the importance of each feature to a model’s predictions.
# MAGIC
# MAGIC AutoML notebooks use the SHAP package to calculate Shapley values. Because these calculations are very memory-intensive, the calculations are not performed by default.
# MAGIC
# MAGIC To calculate and display Shapley values:
# MAGIC
# MAGIC Go to the Feature importance section in an AutoML generated trial notebook.
# MAGIC
# MAGIC 1. Set shap_enabled = True.
# MAGIC 2. Re-run the notebook.
# MAGIC 3. Time series aggregation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time series aggregation
# MAGIC For forecasting problems, when there are multiple values for a timestamp in a time series, AutoML uses the average of the values.
# MAGIC
# MAGIC To use the sum instead, edit the source code notebook. In the Aggregate data by … cell, change .agg(y=(target_col, "avg")) to .agg(y=(target_col, "sum")), as shown:
# MAGIC
# MAGIC Python<br>
# MAGIC group_cols = [time_col] + id_cols <br>
# MAGIC df_aggregation = df_loaded 
# MAGIC   .groupby(group_cols) \
# MAGIC   .agg(y=(target_col, "sum")) \ 
# MAGIC   .reset_index() \
# MAGIC   .rename(columns={ time_col : "ds" })

# COMMAND ----------

# MAGIC %md
# MAGIC #Train ML models with the Databricks AutoML UI
# MAGIC
# MAGIC This article demonstrates how to train a machine learning model using AutoML and the Databricks Machine Learning UI. The AutoML UI steps you through the process of training a classification, regression or forecasting model on a dataset.
# MAGIC
# MAGIC To access the UI:
# MAGIC
# MAGIC 1. Select Machine Learning from the persona switcher at the top of the left sidebar.
# MAGIC 2. In the sidebar, click Create > AutoML Experiment.
# MAGIC
# MAGIC You can also create a new AutoML experiment from the Experiments page.
# MAGIC
# MAGIC The Configure AutoML experiment page displays. On this page, you configure the AutoML process, specifying the dataset, problem type, target or label column to predict, metric to use to evaluate and score the experiment runs, and stopping conditions.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Set up classification or regression problems
# MAGIC You can set up a classification or regression problem using the AutoML UI with the following steps:
# MAGIC
# MAGIC 1. In the Compute field, select a cluster running Databricks Runtime 8.3 ML or above.
# MAGIC
# MAGIC 2. From the ML problem type drop-down menu, select Regression or Classification. If you are trying to predict a continuous numeric value for each observation, such as annual income, select regression. If you are trying to assign each observation to one of a discrete set of classes, such as good credit risk or bad credit risk, select classification.
# MAGIC
# MAGIC 3. Under Dataset, select Browse.
# MAGIC
# MAGIC 4. Navigate to the table you want to use and click Select. The table schema appears.
# MAGIC
# MAGIC For classification and regression problems only, you can specify which columns to include in training and select custom imputation methods.
# MAGIC
# MAGIC 5. Click in the Prediction target field. A drop-down appears listing the columns shown in the schema. Select the column you want the model to predict.
# MAGIC
# MAGIC 6. The Experiment name field shows the default name. To change it, type the new name in the field.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Set up forecasting problems
# MAGIC You can set up a forecasting problem using the AutoML UI with the following steps:
# MAGIC
# MAGIC 1. In the Compute field, select a cluster running Databricks Runtime 10.0 ML or above.
# MAGIC
# MAGIC 2. From the ML problem type drop-down menu, select Forecasting.
# MAGIC
# MAGIC 3. Under Dataset, click Browse. Navigate to the table you want to use and click Select. The table schema appears.
# MAGIC
# MAGIC 4. Click in the Prediction target field. A dropdown menu appears listing the columns shown in the schema. Select the column you want the model to predict.
# MAGIC
# MAGIC 5. Click in the Time column field. A drop-down appears showing the dataset columns that are of type timestamp or date. Select the column containing the time periods for the time series.
# MAGIC
# MAGIC 6. For multi-series forecasting, select the column(s) that identify the individual time series from the Time series identifiers drop-down. AutoML groups the data by these columns as different time series and trains a model for each series independently. If you leave this field blank, AutoML assumes that the dataset contains a single time series.
# MAGIC
# MAGIC 7. In the Forecast horizon and frequency fields, specify the number of time periods into the future for which AutoML should calculate forecasted values. In the left box, enter the integer number of periods to forecast. In the right box, select the units. .. note :: To use Auto-ARIMA, the time series must have a regular frequency (that is, the interval between any two points must be the same throughout the time series). The frequency must match the frequency unit specified in the API call or in the AutoML UI. AutoML handles missing time steps by filling in those values with the previous value.
# MAGIC
# MAGIC 8. In Databricks Runtime 10.5 ML and above, you can save prediction results. To do so, specify a database in the Output Database field. Click Browse and select a database from the dialog. AutoML writes the prediction results to a table in this database.
# MAGIC
# MAGIC 9. The Experiment name field shows the default name. To change it, type the new name in the field.
# MAGIC
# MAGIC ##  Use existing feature tables from Databricks Feature Store
# MAGIC In Databricks Runtime 11.3 LTS ML and above, you can use feature tables in Databricks Feature Store to expand the input training dataset for your classification and regression problems.
# MAGIC
# MAGIC In Databricks Runtime 12.2 LTS ML and above, you can use feature tables in Databricks Feature Store to expand the input training dataset for all of your AutoML problems: classification, regression, and forecasting.
# MAGIC
# MAGIC To create a feature table, see Create a feature table in Databricks Feature Store.
# MAGIC
# MAGIC After you finish configuring your AutoML experiment, you can select a features table with the following steps:
# MAGIC
# MAGIC 1. Click Join features (optional).
# MAGIC
# MAGIC ![](https://docs.databricks.com/_images/automl-join-features.png)
# MAGIC
# MAGIC 2. On the Join Additional Features page, select a feature table in the Feature Table field.
# MAGIC
# MAGIC 3. For each Feature table primary key, select the corresponding lookup key. The lookup key should be a column in the training dataset you provided for your AutoML experiment.
# MAGIC
# MAGIC 4. For time series feature tables, select the corresponding timestamp lookup key. Similarly, the timestamp lookup key should be a column in the training dataset you provided for your AutoML experiment
# MAGIC
# MAGIC ![](https://docs.databricks.com/_images/automl-feature-store-lookup-key.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the experiment and monitor the results
# MAGIC To start the AutoML experiment, click Start AutoML. The experiment starts to run, and the AutoML training page appears. To refresh the runs table, click Refresh button.
# MAGIC
# MAGIC From this page, you can:
# MAGIC
# MAGIC - Stop the experiment at any time.
# MAGIC
# MAGIC - Open the data exploration notebook.
# MAGIC
# MAGIC - Monitor runs.
# MAGIC
# MAGIC - Navigate to the run page for any run.
# MAGIC
# MAGIC  Databricks Runtime 10.1 ML and above, AutoML displays warnings for potential issues with the dataset, such as unsupported column types or high cardinality columns.
# MAGIC
# MAGIC > Note
# MAGIC > Databricks does its best to indicate potential errors or issues. However, this may not be comprehensive and may not capture issues or errors for which you may be searching. Please make sure to conduct your own reviews as well.
# MAGIC
# MAGIC To see any warnings for the dataset, click the Warnings tab on the training page, or on the experiment page after the experiment has completed.
# MAGIC
# MAGIC ![](https://docs.databricks.com/_images/automl-alerts.png)
# MAGIC
# MAGIC When the experiment completes, you can:
# MAGIC
# MAGIC - Register and deploy one of the models with MLflow.
# MAGIC
# MAGIC - Select View notebook for best model to review and edit the notebook that created the best model.
# MAGIC
# MAGIC - Select View data exploration notebook to open the data exploration notebook.
# MAGIC
# MAGIC - Search, filter, and sort the runs in the runs table.
# MAGIC
# MAGIC - See details for any run:
# MAGIC
# MAGIC > - To open the notebook containing source code for a trial run, click in the Source column.
# MAGIC
# MAGIC > - To view results of the run, click in the Models column or the Start Time column. The run page appears showing information about the trial run (such as parameters, metrics, and tags) and artifacts created by the run, including the model. This page also includes code snippets that you can use to make predictions with the model.
# MAGIC
# MAGIC To return to this AutoML experiment later, find it in the table on the Experiments page. The results of each AutoML experiment, including the data exploration and training notebooks, are stored in a databricks_automl folder in the home folder of the user who ran the experiment

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register and deploy a model
# MAGIC You can register and deploy your model with the AutoML UI:
# MAGIC
# MAGIC 1. Select the link in the Models column for the model to register. When a run completes, the best model (based on the primary metric) is the top row.
# MAGIC
# MAGIC      The artifacts section of the run page for the run that created the model displays.
# MAGIC
# MAGIC 2. Select register model button to register the model in Model Registry.
# MAGIC
# MAGIC 3. Select Models Icon Models in the sidebar to navigate to the Model Registry.
# MAGIC
# MAGIC 4. Select the name of your model in the model table. The registered model page displays. From this page, you can serve the model with Serverless Real-Time Inference
# MAGIC  

# COMMAND ----------

