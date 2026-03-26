# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-openai databricks-agents openai-agents=='0.7.0'
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %%writefile model.py
# MAGIC from typing import Generator
# MAGIC
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC )
# MAGIC
# MAGIC
# MAGIC class ResponsesAgentModel(ResponsesAgent):
# MAGIC
# MAGIC     def __init__(self, model: str):
# MAGIC         self.client = WorkspaceClient().serving_endpoints.get_open_ai_client()
# MAGIC         self.model = model
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         messages = [i.model_dump() for i in request.input]
# MAGIC
# MAGIC         response = self.client.chat.completions.create(
# MAGIC             model=self.model,
# MAGIC             messages=self.prep_msgs_for_cc_llm(messages),
# MAGIC         )
# MAGIC
# MAGIC         return ResponsesAgentResponse(
# MAGIC             output=[
# MAGIC                 self.create_text_output_item(
# MAGIC                     text=response.choices[0].message.content, id=response.id
# MAGIC                 )
# MAGIC             ],
# MAGIC         )
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self, request: ResponsesAgentRequest
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         messages = [i.model_dump() for i in request.input]
# MAGIC
# MAGIC         stream = self.client.chat.completions.create(
# MAGIC             model=self.model,
# MAGIC             messages=self.prep_msgs_for_cc_llm(messages),
# MAGIC             stream=True,
# MAGIC         )
# MAGIC
# MAGIC         full_message = ""
# MAGIC         msg_id = ""
# MAGIC
# MAGIC         for chunk in stream:
# MAGIC             chunk_dict = chunk.to_dict()
# MAGIC
# MAGIC             if not chunk_dict["choices"]:
# MAGIC                 continue
# MAGIC
# MAGIC             delta = chunk_dict["choices"][0]["delta"]
# MAGIC             msg_id = chunk_dict["id"]
# MAGIC             content = delta.get("content", "")
# MAGIC
# MAGIC             if content:
# MAGIC                 full_message += content
# MAGIC                 yield ResponsesAgentStreamEvent(
# MAGIC                     **self.create_text_delta(delta=content, item_id=msg_id),
# MAGIC                 )
# MAGIC             
# MAGIC         yield ResponsesAgentStreamEvent(
# MAGIC             type="response.output_item.done",
# MAGIC             item=self.create_text_output_item(
# MAGIC                 text=full_message,
# MAGIC                 id=msg_id,
# MAGIC             ),
# MAGIC         )
# MAGIC
# MAGIC         return
# MAGIC
# MAGIC
# MAGIC # TODO: Change this to the name of the Databricks model serving endpoint you want to use.
# MAGIC # You can find available foundation model endpoints in your Databricks workspace under
# MAGIC # the AI Gateway section.
# MAGIC BASE_ENDPOINT = "databricks-llama-4-maverick"
# MAGIC
# MAGIC # TODO: Change this prefix to something unique to you (e.g. your name or team name).
# MAGIC # It is used to name the MLflow experiment, the registered model, and the serving endpoint.
# MAGIC MODEL_PREFIX = "mn-demo"
# MAGIC MODEL_NAME = MODEL_PREFIX + "-model"
# MAGIC
# MAGIC mlflow.openai.autolog()
# MAGIC mlflow.set_tracking_uri("databricks")
# MAGIC mlflow.set_experiment("/Shared/" + MODEL_NAME)
# MAGIC mlflow.models.set_model(ResponsesAgentModel(BASE_ENDPOINT))

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import logging

import mlflow
from mlflow.models.resources import DatabricksServingEndpoint
from mlflow.types.responses import ResponsesAgentRequest

from model import BASE_ENDPOINT, MODEL_NAME, MODEL_PREFIX

UC_LOCATION = f"workspace.default.{MODEL_NAME}"

logging.getLogger("mlflow").setLevel(logging.ERROR)

example = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the fibonacci sequence"},
]

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name=MODEL_NAME,
        python_model="model.py",
        input_example=ResponsesAgentRequest(input=example),
        registered_model_name=UC_LOCATION,
        resources=[DatabricksServingEndpoint(endpoint_name=BASE_ENDPOINT)],
    )

# COMMAND ----------

from databricks import agents

ENDPOINT_NAME = MODEL_PREFIX + "-endpoint"

agents.deploy(
    UC_LOCATION,
    scale_to_zero=True,
    model_version=logged_agent_info.registered_model_version,
    endpoint_name=ENDPOINT_NAME,
)