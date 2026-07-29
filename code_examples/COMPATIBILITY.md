# Code-example compatibility register

The repository intentionally spans multiple frameworks and publication dates, so one global requirements file would be misleading. This register records examples that are known to use historical APIs or provider-specific identifiers. It is not an exhaustive dependency lockfile.

| Area | Affected examples | Status and recommended action |
|---|---|---|
| Weaviate Python client | `Part_06_Chapter_6.2B_client_initialization_and_authentication_code_01_client_initialization_and_authentication.py` and adjacent schema/batch snippets | Uses the deprecated v3 client style. Migrate to the v4 collections API before using a current client. |
| Semantic Kernel planning | `Part_02_Chapter_2.5_Semantic_Kernel_code_06_function_calling_planner.py`, `...code_12_plugin_orchestration_setup.py` | `FunctionCallingStepwisePlanner` was removed. Use automatic function calling and current execution settings. |
| LangChain memory | `Part_02_Chapter_2.3_LangChain_code_03_configure_conversation_memory.py` and its study-plan references | Demonstrates the older `ConversationBufferMemory` pattern. Current agents manage short-term memory in agent state and persist it through a checkpointer. |
| LangChain integration with NVIDIA NIM | `Part_07_Chapter_7.3_agent_initialization_code_01_agent_initialization.py` and adjacent profiling excerpts | Uses the older `langchain.llms.NIM` and `AgentExecutor` pattern. Current NVIDIA guidance uses `langchain-nvidia-ai-endpoints.ChatNVIDIA` with a LangGraph tool-calling agent; migrate the surrounding sequence together before execution. |
| MLflow registry lifecycle | `Part_04_Chapter_4.1C_22_mlflow_production_transition.sh` | Updated in this repository to use a registered-model alias instead of deprecated model stages. |
| NVIDIA NIM images | Examples containing `nvcr.io/nvidia/nim...` or unversioned `:latest` tags | Image names are model- and release-specific. Use the current NIM support matrix and an exact `nvcr.io/nim/<publisher>/<model>:<tag>` image. Do not assume the historical sample image exists. |
| General framework imports | Many examples | Create a chapter-specific environment and pin versions after consulting current upstream documentation. Do not infer compatibility solely because a file parses. |

## Primary migration references

- Weaviate Python client: <https://docs.weaviate.io/weaviate/client-libraries/python>
- Semantic Kernel planner migration: <https://learn.microsoft.com/semantic-kernel/support/migration/stepwise-planner-migration-guide>
- LangChain short-term memory: <https://docs.langchain.com/oss/python/langchain/short-term-memory>
- NVIDIA NIM with LangChain and LangGraph: <https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/tool-calling-and-mcp.html>
- MLflow model-registry workflows: <https://mlflow.org/docs/latest/ml/model-registry/workflow/>
- NVIDIA NIM installation and image selection: <https://docs.nvidia.com/nim/large-language-models/latest/get-started/installation.html>
