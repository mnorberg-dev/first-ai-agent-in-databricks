# first-ai-agent-in-databricks

A minimal example of building and deploying an AI agent in Databricks using MLflow's `ResponsesAgent` interface.

## Setup

### 1. Clone this repo into your Databricks workspace

In your Databricks workspace, open the left sidebar and navigate to **Workspace**. Then:

1. Click **Add** > **Git folder**
2. Paste in this repository's URL
3. Click **Create Git folder**

> **Note:** You may need to link a Git provider account first. Go to your Databricks user settings (**Settings > Linked accounts**) and connect your GitHub (or other Git provider) account before cloning.

The notebooks will appear as runnable files directly in your workspace.

### 2. Configure your constants

Before running the notebooks, update the following constants in `src/agent-model.py`:

| Constant | Where | What to set |
|---|---|---|
| `BASE_ENDPOINT` | `agent-model.py` cell 1 (`model.py` section) | The name of the Databricks model serving endpoint to use as the underlying LLM. Find available endpoints under **Serving > Endpoints** in your workspace. |
| `MODEL_PREFIX` | `agent-model.py` cell 1 (`model.py` section) | A unique prefix for your resources (e.g. your name or team). Used to name the MLflow experiment, registered model, and serving endpoint. |

If you are running `src/llm-judges.py`, also update:

| Constant          | Where           | What to set                                                                              |
| ----------------- | --------------- | ---------------------------------------------------------------------------------------- |
| `EXPERIMENT_NAME` | `llm-judges.py` | Must match `/Shared/<MODEL_PREFIX>-model` — the experiment path set in `agent-model.py`. |

### 3. Run the notebooks in order

1. **`src/agent-model.py`** — Writes `model.py`, logs the agent to MLflow, registers it in Unity Catalog, and deploys it as a serving endpoint.
2. **Use the agent** — Interact with it via the AI Playground, REST API, SQL, or directly in Python (see below). This generates traces in MLflow.
3. **`src/llm-judges.py`** _(optional)_ — Pulls traces from the MLflow experiment and evaluates them using LLM judges. Run this only after you have used the agent, otherwise there are no traces to evaluate.

---

## Using the deployed agent

Once `agent-model.py` has run successfully, your agent is available as a Databricks model serving endpoint named `<MODEL_PREFIX>-endpoint`.

### Call it via the REST API

```bash
curl -X POST https://<your-workspace>.azuredatabricks.net/serving-endpoints/<MODEL_PREFIX>-endpoint/invocations \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the fibonacci sequence?"}
    ]
  }'
```

### Use it in the AI Playground

1. In your Databricks workspace, navigate to **Machine Learning > Playground**
2. Click the model selector at the top and choose your endpoint (`<MODEL_PREFIX>-endpoint`) from the list of serving endpoints
3. Start chatting

### Call it with `ai_query` in SQL

You can call the agent directly from a SQL cell or notebook using the `ai_query` built-in function:

```sql
SELECT ai_query(
  '<MODEL_PREFIX>-endpoint',
  'What is the fibonacci sequence?'
)
```

You can also use it over a table column:

```sql
SELECT
  question,
  ai_query('<MODEL_PREFIX>-endpoint', question) AS answer
FROM my_table
```

### Call it directly in Python using `ResponsesAgentModel`

You can instantiate and call the agent class directly without going through a serving endpoint:

```python
from mlflow.types.responses import ResponsesAgentRequest
from model import ResponsesAgentModel, BASE_ENDPOINT

agent = ResponsesAgentModel(BASE_ENDPOINT)

request = ResponsesAgentRequest(input=[
    {"role": "user", "content": "What is the fibonacci sequence?"}
])

# Non-streaming
response = agent.predict(request)
print(response.output[0].content)

# Streaming
for event in agent.predict_stream(request):
    print(event)
```

### Call it with MLflow deployments

```python
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
response = client.predict(
    endpoint="<MODEL_PREFIX>-endpoint",
    inputs={
        "messages": [
            {"role": "user", "content": "What is the fibonacci sequence?"}
        ]
    },
)
print(response)
```
