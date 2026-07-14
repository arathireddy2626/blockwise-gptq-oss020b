import sglang as sg
import time
import json

@sg.function
def gen(s, prompt):
    s += prompt
    s += sg.gen("output", max_tokens=80)

prompts = ["The capital of France is", "2 + 2 ="]

# Load model
runtime = sg.Runtime(
    model_path="models/gpt-oss-20b-BF16",
    tp_size=1,
)
sg.set_default_backend(runtime)

# Benchmark
for prompt in prompts:
    result = gen(prompt=prompt)
    print(f"{prompt}: {result['output']}")

runtime.shutdown()
