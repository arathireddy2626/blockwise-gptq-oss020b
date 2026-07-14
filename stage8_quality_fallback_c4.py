from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

model_path = 'models/gpt-oss-20b-BF16-NVFP4-modelopt'  # original C4-calibrated model

QUALITY_PROMPTS = [
    'The capital of France is',
    '2 + 2 =',
    'def fibonacci(n):\n    ',
    'The largest planet in the solar system is',
    'The speed of light in a vacuum is approximately',
    'In Python, a list comprehension that squares numbers 1-10 is:'
]

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print(f'Loading NVFP4 quantized model from {model_path}...')
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True
)

results = []
for prompt in QUALITY_PROMPTS:
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append({'prompt': prompt, 'output': text})
    print(f'PROMPT: {prompt}')
    print(f'OUTPUT: {text}')
    print('-' * 70)

with open('results/stage8_quality_fallback_c4.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved.')
