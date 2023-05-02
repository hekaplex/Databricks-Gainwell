# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # Buy Now, Pay Later (BNPL)
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/fsi/credit_decisioning/fsi-credit-decisioning-ml-3.png" style="float: right" width="800px">
# MAGIC
# MAGIC
# MAGIC ### *"Can we allow a user without sufficient balance in their account to complete the current (debit or credit) transaction?"*
# MAGIC
# MAGIC
# MAGIC We will utilize our credit risk model (built in the previous steps) in real-time to answer this question.
# MAGIC
# MAGIC The payment system will be able to call our API in realtime and get a score within a few ms. 
# MAGIC
# MAGIC With this information, we'll be able to offer our customer the choice to pay with a credit automatically, or refuse if the model believes the risk is too high and  will likely result in a payment default.
# MAGIC
# MAGIC
# MAGIC These types of decisions are typically embedded in live Point Of Sales (stores, online shop). That is why we need real-time serving capabilities.

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploying the Credit Scoring model for real-time serving
# MAGIC
# MAGIC
# MAGIC Let's deploy our model behind a scalable API to evaluate credit-worthiness in real-time.
# MAGIC
# MAGIC ## Databricks Model Serving
# MAGIC
# MAGIC Now that our model has been created with Databricks AutoML, we can easily flag it as Production Ready and turn on Databricks Model Serving.
# MAGIC
# MAGIC We'll be able to send HTTP Requests and get inference in real-time.
# MAGIC
# MAGIC Databricks Model Serving is fully serverless:
# MAGIC
# MAGIC * One-click deployment. Databricks will handle scalability, providing blazing fast inferences and startup time.
# MAGIC * Scale down to zero as an option for best TCO (will shut down if the endpoint isn't used).
# MAGIC * Built-in support for multiple models & version deployed.
# MAGIC * A/B Testing and easy upgrade, routing traffic between each versions while measuring impact.
# MAGIC * Built-in metrics & monitoring.

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false $catalog=dbdemos $db=fsi_credit_decisioning

# COMMAND ----------

# DBTITLE 1,Make sure our last model version is deployed in production in our registry
client = mlflow.tracking.MlflowClient()
models = client.get_latest_versions("dbdemos_fsi_credit_decisioning")
models.sort(key=lambda m: m.version, reverse=True)
latest_model = models[0]

if latest_model.current_stage != "Production":
    client.transition_model_version_stage(
        "dbdemos_fsi_credit_decisioning",
        latest_model.version,
        stage="Production",
        archive_existing_versions=True
    )

# COMMAND ----------

# DBTITLE 1,Deploy the model in our serving endpoint
#See helper in companion notebook
serving_client = EndpointApiClient()

serving_client.create_enpoint_if_not_exists(
    "dbdemos_fsi_credit_decisioning_endpoint",
    model_name="dbdemos_fsi_credit_decisioning",
    model_version=latest_model.version,
    workload_size="Small",
    scale_to_zero_enabled=True,
    wait_start=True
)

# COMMAND ----------

# MAGIC %md 
# MAGIC Our model endpoint was automatically created. 
# MAGIC
# MAGIC Open the [endpoint UI](#mlflow/endpoints/dbdemos_fsi_credit_decisioning_endpoint) to explore your endpoint and use the UI to send queries.
# MAGIC
# MAGIC *Note that the first deployment will build your model image and take a few minutes. It'll then stop & start instantly.*

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Testing the model
# MAGIC
# MAGIC Now that the model is deployed, let's test it with information for a customer trying to temporarily increase their credit limits or obtain a micro-loan at the point-of-sale.

# COMMAND ----------

p = ModelsArtifactRepository("models:/dbdemos_fsi_credit_decisioning/Production").download_artifacts("") 
dataset =  {"dataframe_split": Model.load(p).load_input_example(p).to_dict(orient='split')}

# COMMAND ----------

from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

endpoint_url = f"{serving_client.base_url}/realtime-inference/dbdemos_fsi_credit_decisioning_endpoint/invocations"
inferences = requests.post(
    endpoint_url, json=dataset, headers=serving_client.headers
).json()

prediction_score = list(inferences["predictions"])[0]
print(
    f"The transaction will be approved. Score: {prediction_score}."
    if prediction_score == 0
    else f"The transaction will not be approved. Score: {prediction_score}."
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Updating your model and monitoring its performance with A/B testing 
# MAGIC
# MAGIC
# MAGIC <img style="float: right;" src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/fsi/fraud-detection/model-serving-ab-testing.png" width="850px" />
# MAGIC
# MAGIC Databricks Model Serving let you easily deploy & test new versions of your model.
# MAGIC
# MAGIC You can dynamically reconfigure your endpoint to route a subset of your traffic to a newer version. In addition, you can leverage endpoint monitoring to understand your model behavior and track your A/B deployment.
# MAGIC
# MAGIC * Without making any production outage
# MAGIC * Slowly routing requests to the new model
# MAGIC * Supporting auto-scaling & potential bursts
# MAGIC * Performing some A/B testing ensuring the new model is providing better outcomes
# MAGIC * Monitorig our model outcome and technical metrics (CPU/load etc)
# MAGIC
# MAGIC Databricks makes this process super simple with Serverless Model Serving endpoint.
# MAGIC
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=4994874874414839&aip=1&t=event&ec=dbdemos&ea=VIEW&dp=%2F_dbdemos%2Flakehouse%2Flakehouse-fsi-credit%2F03-Data-Science-ML%2F03.4-model-serving-BNPL-credit-decisioning&uid=6165859300088079">

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Model monitoring and A/B testing analysis
# MAGIC
# MAGIC Because the Model Serving runs within our Lakehouse, Databricks will automatically save and track all our Model Endpoint results as a Delta Table.
# MAGIC
# MAGIC We can then easily plug a feedback loop to start analysing the revenue in $ each model is offering. 
# MAGIC
# MAGIC All these metrics, including A/B testing validation (p-values etc) can then be pluged into a Model Monitoring Dashboard and alerts can be sent for errors, potentially triggering new model retraining or programatically updating the Endpoint routes to fallback to another model.
# MAGIC
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/fsi/fraud-detection/model-serving-monitoring.png" width="1200px" />

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Next
# MAGIC
# MAGIC Making sure your model is fair towards customers of any demographics are extremely important parts of building production-ready ML models for FSI use cases. <br/>
# MAGIC Explore your model with [03.5-Explainability-and-Fairness-credit-decisioning]($./03.5-Explainability-and-Fairness-credit-decisioning) on the Lakehouse.
# MAGIC
# MAGIC ## Conclusion: the power of the Lakehouse
# MAGIC
# MAGIC In this demo, we've seen an end 2 end flow with the Lakehouse:
# MAGIC
# MAGIC - Data ingestion made simple with Delta Live Table
# MAGIC - Leveraging Databricks warehouse to making credit decisions
# MAGIC - Model Training with AutoML for citizen Data Scientist
# MAGIC - Ability to tune our model for better results, improving our revenue
# MAGIC - Ultimately, the ability to deploy and make explainable ML predictions, made possible with the full Lakehouse capabilities.
# MAGIC
# MAGIC [Go back to the introduction]($../00-Credit-Decisioning) or discover how to use Databricks Workflow to orchestrate everything together through the [05-Workflow-Orchestration-credit-decisioning]($../05-Workflow-Orchestration/05-Workflow-Orchestration-credit-decisioning).