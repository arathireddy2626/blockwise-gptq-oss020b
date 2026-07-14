import torch
import time
import json
from pathlib import Path
from vllm import LLM, SamplingParams

def benchmark_model(model_path, model_name):
    print(f"\n{'='*60}")
    print(f"Benchmarking {model_name}: {model_path}")
    print(f"{'='*60}")
    
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.88,
        enforce_eager=True,
        disable_log_stats=True,
    )
    
    prompts = [
        "The capital of France is",
        "2 + 2 =",
        "def fibonacci(n):\n    ",
        "The largest planet in the solar system is",
    ]
    
    params = SamplingParams(temperature=0.0, max_tokens=80)
    
    # Warmup
    llm.generate(prompts[:1], params)
    torch.cuda.synchronize()
    
    # Quality test
    print("\n[Quality] Running prompts...")
    outputs = llm.generate(prompts, params)
    for i, out in enumerate(outputs):
        print(f"  Prompt {i}: {out.outputs[0].text[:60]}...")
    
    # TTFT
    print("\n[TTFT] Measuring time-to-first-token...")
    t0 = time.perf_counter()
    for _ in range(5):
        llm.generate([prompts[0]], SamplingParams(temperature=0.0, max_tokens=1))
        torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"  TTFT: {ttft_ms:.2f} ms")
    
    # Throughput at batch sizes
    print("\n[Throughput] Running batches...")
    results = {"ttft_ms": ttft_ms, "throughput": []}
    for bs in [1, 4, 8]:
        batch = prompts[:bs]
        t0 = time.perf_counter()
        for _ in range(3):
            outs = llm.generate(batch, params)
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / 3
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        tps = total_tokens / elapsed
        results["throughput"].append({
            "batch_size": bs,
            "tokens_per_second": tps,
            "elapsed_s": elapsed
        })
        print(f"  Batch {bs}: {tps:.1f} tok/s ({elapsed:.2f}s)")
    
    return results

if __name__ == "__main__":
    results = {}
    
    print("\nLoading BF16 model...")
    results["bf16"] = benchmark_model("models/gpt-oss-20b-BF16", "BF16")
    
    print("\n\nLoading NVFP4 model...")
    results["nvfp4"] = benchmark_model("models/gpt-oss-20b-BF16-NVFP4-modelopt", "NVFP4")
    
    # Speedup
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    bf16_ttft = results["bf16"]["ttft_ms"]
    nvfp4_ttft = results["nvfp4"]["ttft_ms"]
    print(f"TTFT speedup: {bf16_ttft/nvfp4_ttft:.2f}x")
    
    for bs_idx in range(len(results["bf16"]["throughput"])):
        bf16_tps = results["bf16"]["throughput"][bs_idx]["tokens_per_second"]
        nvfp4_tps = results["nvfp4"]["throughput"][bs_idx]["tokens_per_second"]
        bs = results["bf16"]["throughput"][bs_idx]["batch_size"]
        speedup = nvfp4_tps / bf16_tps
        print(f"Batch {bs} throughput speedup: {speedup:.2f}x")
    
    # Save results
    with open("results/stage8_benchmark_simple.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/stage8_benchmark_simple.json")
