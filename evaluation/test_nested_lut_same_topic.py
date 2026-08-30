#!/usr/bin/env python3
"""
Test nested LUT with same-theme prompts to observe cold/hot GPU cache effects.

Uses subprocess isolation between tiers to ensure clean GPU cleanup.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET = "/home/hh/zip_Moe/LUT_MoE/evaluation/dataset/sharegpt_gpt4.jsonl"
RESULTS_DIR = "/home/hh/zip_Moe/LUT_MoE/evaluation/results"

TOPICS = {
    "programming": [
        "python", "javascript", "java", "c++", "rust", "golang", "sql", "mysql",
        "postgres", "api", "code", "function", "bug", "error", "framework",
        "react", "vue", "angular", "node.js", "django", "flask", "docker",
        "kubernetes", "git", "github", "linux", "bash", "shell", "regex",
        "database", "redis", "mongodb", "html", "css", "typescript",
        "programming", "developer", "software", "algorithm", "compiler",
        "debug", "deploy", "server", "aws", "azure", "lambda",
        "tensorflow", "pytorch", "machine learning", "neural network",
        "how to use", "how to implement", "how do i", "setup", "install",
        "syntax", "compile", "runtime", "memory leak",
    ],
    "writing": [
        "write", "article", "essay", "blog", "content", "story", "novel",
        "poem", "poetry", "creative writing", "copywriting", "summary",
        "summarize", "paraphrase", "rewrite", "edit", "proofread",
    ],
    "math_science": [
        "math", "equation", "physics", "chemistry", "biology", "theorem",
        "calculus", "algebra", "statistics", "probability", "signal",
        "quantum", "molecular", "genetic", "formula", "proof",
    ],
}


def classify_prompt(text):
    text_lower = text.lower()
    scores = {}
    for topic, keywords in TOPICS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[topic] = score
    return max(scores, key=scores.get) if scores else "other"


def find_same_topic_prompts(num_prompts=8, topic="programming", min_score=2):
    matching = []
    with open(DATASET) as f:
        for line in f:
            d = json.loads(line)
            convs = d.get("conversations", d.get("conversation", []))
            if not convs:
                continue
            first_msg = convs[0].get("value", "")
            if not first_msg:
                continue
            if classify_prompt(first_msg) == topic:
                score = sum(1 for kw in TOPICS[topic] if kw in first_msg.lower())
                if score >= min_score:
                    matching.append(first_msg)
    if len(matching) < num_prompts:
        return matching
    step = len(matching) // num_prompts
    return [matching[i * step] for i in range(num_prompts)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--topic", type=str, default="programming")
    parser.add_argument("--tiers", type=str, default="0,1,2")
    parser.add_argument("--code_type", type=str, default="NESTEDLUT")
    args = parser.parse_args()

    tiers = [int(t) for t in args.tiers.split(",")]

    print(f"Finding {args.num_prompts} same-topic prompts (topic: {args.topic})...")
    prompts = find_same_topic_prompts(args.num_prompts, args.topic)
    if len(prompts) < args.num_prompts:
        print(f"WARNING: Only found {len(prompts)} matching prompts")
    prompts = prompts[:args.num_prompts]

    print(f"Selected {len(prompts)} prompts:")
    for i, p in enumerate(prompts):
        print(f"  [{i}] {p[:120]}...")

    # Save prompts to temp file for subprocess
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(prompts, f)
        prompts_file = f.name

    runner_script = os.path.join(os.path.dirname(__file__), "_nested_lut_tier_runner.py")
    results = {}
    tier_outputs = {}

    for tier in tiers:
        print(f"\n{'='*60}")
        print(f"  Tier {tier}: ", end="")
        if tier == 0:
            print("full 256 (8-bit)")
        elif tier == 1:
            print("mapped 64 (6-bit)")
        else:
            print("mapped 16 (4-bit)")
        print(f"{'='*60}")

        out_file = os.path.join(RESULTS_DIR, f"nested_lut_tier_{tier}_{int(time.time())}.json")

        print(f"  Launching subprocess for tier {tier}...")
        t0 = time.perf_counter()
        log_file = out_file.replace(".json", ".log")
        with open(log_file, "w") as logf:
            result = subprocess.run(
                [sys.executable, runner_script,
                 "--tier", str(tier),
                 "--code_type", args.code_type,
                 "--prompts_file", prompts_file,
                 "--max_prompt_length", str(args.max_prompt_length),
                 "--output", out_file],
                stdout=logf, stderr=subprocess.STDOUT, text=True, timeout=900,
            )
        # Print last few lines of log
        with open(log_file) as logf:
            lines = logf.readlines()
            for line in lines[-15:]:
                print(f"  | {line.rstrip()}")
        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            print(f"ERROR: Tier {tier} subprocess failed (exit code {result.returncode})!")
            print(f"       Log file: {log_file}")
            continue

        print(f"  Tier {tier} completed in {elapsed:.1f}s (wall clock)")

        if os.path.exists(out_file):
            with open(out_file) as f:
                tier_outputs[tier] = json.load(f)
        else:
            print(f"  WARNING: No output file found for tier {tier}")

    os.unlink(prompts_file)

    # ── Cross-tier comparison ──
    print(f"\n{'='*60}")
    print("  Cross-Tier Comparison")
    print(f"{'='*60}")
    print(f"  {'Tier':<8} {'Load(s)':<10} {'Cold(s)':<10} {'Hot_avg(s)':<12} {'Speedup':<10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    tier_names = {0: "256(8b)", 1: "64(6b)", 2: "16(4b)"}

    # Merge results
    merged = {}
    for tier in tiers:
        r = tier_outputs.get(tier, {})
        if not r:
            merged[str(tier)] = {"error": "no data"}
            continue
        name = tier_names.get(tier, str(tier))
        print(f"  {name:<8} {r['load_time']:<10.1f} {r['cold_time']:<10.3f} "
              f"{r.get('hot_avg', 0):<12.3f} {r.get('speedup', 0):<10.2f}x")
        merged[str(tier)] = r

    merged["config"] = {
        "topic": args.topic,
        "num_prompts": len(prompts),
        "prompts": prompts,
        "tiers_tested": tiers,
        "code_type": args.code_type,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(RESULTS_DIR, f"nested_lut_test-{timestamp}.json")
    with open(result_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
