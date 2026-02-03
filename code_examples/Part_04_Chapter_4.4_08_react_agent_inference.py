# agent_inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import nvtx  # NVIDIA Tools Extension for custom ranges

# Load model with GPU acceleration
model_name = "meta-llama/Llama-3.1-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def search_tool(query: str) -> str:
    """Simulated search tool with realistic latency"""
    with nvtx.annotate("search_tool", color="blue"):
        import time
        time.sleep(0.15)  # Simulate 150ms API call
        return f"Search results for: {query}"

def react_agent_step(prompt: str) -> tuple[str, str]:
    """Single ReAct reasoning step with NVTX annotations"""

    # Thought generation
    with nvtx.annotate("llm_thought_generation", color="green"):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )
        thought = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Action execution
    if "search(" in thought:
        query = thought.split("search(")[1].split(")")[0]
        observation = search_tool(query)
    else:
        observation = "No action required"

    return thought, observation

# Profile 10 reasoning steps
with nvtx.annotate("react_agent_workflow", color="red"):
    for i in range(10):
        with nvtx.annotate(f"step_{i}", color="yellow"):
            thought, obs = react_agent_step(
                f"Question: What is the capital of France? Step {i}"
            )
