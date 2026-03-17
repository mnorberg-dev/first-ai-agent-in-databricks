# Databricks notebook source
# MAGIC %pip install -U -qqqq mlflow databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import logging

import mlflow
from mlflow import MlflowClient
from mlflow.genai.scorers import Guidelines, RelevanceToQuery, Safety

# TODO: Set this to match the experiment path set in agent-model.py ("/Shared/" + MODEL_NAME).
EXPERIMENT_NAME = "/Shared/mn-demo-model"

# Client configuration
client = MlflowClient()

# Set experiment
mlflow.set_experiment(EXPERIMENT_NAME)

# Gather experiment id
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
experiment_id = experiment.experiment_id

# Logger setup
logging.getLogger("mlflow").setLevel(logging.ERROR)

# Traces collection
traces = client.search_traces(max_results=50, locations=[experiment_id])

# Scorer setup
scorers = [
    Guidelines(
        name="Concise", guidelines="The response must be concise and to the point."
    ),
    RelevanceToQuery(),
    Safety(),
]

# Evaluation
eval_results = mlflow.genai.evaluate(
    data=traces,
    scorers=scorers,
)