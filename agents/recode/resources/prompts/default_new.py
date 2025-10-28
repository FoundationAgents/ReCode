EXPAND_PROMPT = """
You are the EXPAND step in the LLM Agent loop. You need to replace the current placeholder function node with its code implementation.

Decide how to implement the placeholder:
- If the subtask of current function can be done in 1-2 primitive actions from the list below, write them directly using `run(action: str)`.
- If it will take more than 2 primitive actions, instead break it into smaller placeholder functions. Each sub-goal should be clear, meaningful, and ordered so that completing them achieves the current task.

All legal primitive actions are:
{available_actions}
And all of them should be used in the function `run(action: str) -> str`, which returns an observation in string format.

All the placeholder functions should be used in the format: var_out1, var_out2, ... = snake_style_function_name(var_in1, var_in2="explicitly declared variables will also be registered", ...), in which the function name should explicitly represents the subtask you are going to take.

Do not invent or guess any details that are not present in the provided variables. If essential information is missing or uncertain (such as which target to use, what value to set, or which step to take next), write a descriptive placeholder function that explicitly represents the missing decision), to be expanded later.
Do not assume that any condition or prerequisite is already met unless explicitly confirmed. If something must be prepared, accessed, or changed, include explicit steps or sub-goals to do so.

In your response:
1. Start with a brief natural language explanation of how you will complete or break down the task, encluded with <think> and </think>.
2. Then output a Python code with <execute> and </execute> tags, containing only valid actions or commands for this environment. Do not create functions with `def`, and do not place placeholder functions inside loop or condition structures.

---
Here are some examples to guide the style and format, each example is ONLY ONE turn of the interaction:
{examples}
(End of Examples)
---

The current function to expand is:
{task}
The variables you can use is:
{variables}
"""