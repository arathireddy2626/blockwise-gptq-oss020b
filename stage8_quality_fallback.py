"""
Stage 8 Quality Fallback Test
==============================
Since vLLM/SGLang can't run on this Blackwell GPU (FlashInfer/sgl_kernel
sm120 incompatibility), this tests model quality directly via transformers,
covering the same prompts Stage 8 normally uses.

This does NOT measure throughput/TTFT (requires vLLM's batching engine),
but DOES confirm the quantized model generates coherent, correct text.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import time

model_path = 'models/gpt-oss-20b-BF16-NVFP4-wikitext2-fixed'

QUALITY_PROMPTS = [
    'The capital of France is',
    '2 + 2 =',
    'def fibonacci(n):\n    ',
    'The largest planet in the solar system is',
    'The speed of light in a vacuum is approximately',
    'In Python, a list comprehension that squares numbers 1-10 is:'
]

THROUGHPUT_PROMPTS = [
    'Explain the theory of general relativity in simple terms.',
    'What are the main differences between supervised and unsupervised learning?',
    'Write a Python function that checks if a string is a palindrome.',
    'Describe the water cycle and its importance to the ecosystem.',
    'What is the difference between a list and a tuple in Python?',
    'Explain how transformers work in natural language processing.',
    'What are the SOLID principles in software engineering?',
    'Write a SQL query to find the top 5 highest-paid employees.'
]

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print(f'Loading NVFP4 quantized model from {model_path}...')
t0 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True
)
load_time = time.perf_counter() - t0
print(f'Model loaded in {load_time:.1f}s\n')

results = {
    'model_path': model_path,
    'load_time_s': load_time,
    'quality_prompts': [],
    'throughput_prompts': []
}

print('=' * 70)
print('QUALITY PROMPTS (short factual/code)')
print('=' * 70)
for prompt in QUALITY_PROMPTS:
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    t0 = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    gen_time = time.perf_counter() - t0
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results['quality_prompts'].append({
        'prompt': prompt,
        'output': text,
        'gen_time_s': gen_time
    })
    print(f'PROMPT: {prompt}')
    print(f'OUTPUT: {text}')
    print(f'[{gen_time:.2f}s]')
    print('-' * 70)

print('\n' + '=' * 70)
print('THROUGHPUT PROMPTS (longer explanatory)')
print('=' * 70)
for prompt in THROUGHPUT_PROMPTS:
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    t0 = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    gen_time = time.perf_counter() - t0
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results['throughput_prompts'].append({
        'prompt': prompt,
        'output': text,
        'gen_time_s': gen_time
    })
    print(f'PROMPT: {prompt}')
    print(f'OUTPUT: {text[:200]}...' if len(text) > 200 else f'OUTPUT: {text}')
    print(f'[{gen_time:.2f}s]')
    print('-' * 70)

output_path = 'results/stage8_quality_fallback_wikitext2_fixed.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nAll results saved to {output_path}')
print(f'Total prompts tested: {len(QUALITY_PROMPTS) + len(THROUGHPUT_PROMPTS)}')
