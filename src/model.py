from typing import Generator

from databricks.sdk import WorkspaceClient

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)


class ResponsesAgentModel(ResponsesAgent):

    def __init__(self, model: str):
        self.client = WorkspaceClient().serving_endpoints.get_open_ai_client()
        self.model = model

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        messages = [i.model_dump() for i in request.input]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.prep_msgs_for_cc_llm(messages),
        )

        return ResponsesAgentResponse(
            output=[
                self.create_text_output_item(
                    text=response.choices[0].message.content, id=response.id
                )
            ],
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        messages = [i.model_dump() for i in request.input]

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.prep_msgs_for_cc_llm(messages),
            stream=True,
        )

        full_message = ""
        msg_id = ""

        for chunk in stream:
            chunk_dict = chunk.to_dict()

            if not chunk_dict["choices"]:
                continue

            delta = chunk_dict["choices"][0]["delta"]
            msg_id = chunk_dict["id"]
            content = delta.get("content", "")

            if content:
                full_message += content
                yield ResponsesAgentStreamEvent(
                    **self.create_text_delta(delta=content, item_id=msg_id),
                )
            
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(
                text=full_message,
                id=msg_id,
            ),
        )

        return


# TODO: Change this to the name of the Databricks model serving endpoint you want to use.
# You can find available foundation model endpoints in your Databricks workspace under
# Serving > Endpoints, or use any custom endpoint you have deployed.
BASE_ENDPOINT = "databricks-llama-4-maverick"

# TODO: Change this prefix to something unique to you (e.g. your name or team name).
# It is used to name the MLflow experiment, the registered model, and the serving endpoint.
MODEL_PREFIX = "mn-demo"
MODEL_NAME = MODEL_PREFIX + "-model"

mlflow.openai.autolog()
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/" + MODEL_NAME)
mlflow.models.set_model(ResponsesAgentModel(BASE_ENDPOINT))
