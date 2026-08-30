#!/usr/bin/env python3
"""Fix all CUDA version mismatches for vLLM."""
import os, subprocess, shutil

PT = os.path.expanduser("~/.local/bin/patchelf")

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

# Step 1: Patch the CUDA 12.8 runtime to have SONAME libcudart.so.13
CUDA_RT = "/home/hh/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
FIXED_RT = "/home/hh/.local/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13"
shutil.copy2(CUDA_RT, FIXED_RT)
run([PT, "--set-soname", "libcudart.so.13", FIXED_RT])
# Add libcudart.so.13 as version definition
run([PT, "--add-needed", "libcudart.so.13", FIXED_RT])

# Step 2: Revert the vLLM .so changes (put back libcudart.so.13 as NEEDED)
VLLM = "/home/hh/.local/lib/python3.12/site-packages/vllm"
for f in sorted(os.listdir(VLLM)):
    if not f.endswith(".abi3.so"):
        continue
    fp = os.path.join(VLLM, f)
    run([PT, "--replace-needed", "libcudart.so.12", "libcudart.so.13", fp])

# Step 3: Clear symbol versions on vLLM .so files (so they don't check version)
for f in sorted(os.listdir(VLLM)):
    if not f.endswith(".abi3.so"):
        continue
    fp = os.path.join(VLLM, f)
    r = run(["objdump", "-T", fp])
    for line in r.stdout.split("\n"):
        if "libcudart.so.13" in line and "DF *UND*" in line:
            sym = line.strip().split()[-1]
            run([PT, "--clear-symbol-version", sym, fp])

print("All fixes applied. Testing import...")
r = run(["env", "LD_LIBRARY_PATH=/home/hh/.local/lib/python3.12/site-packages/nvidia/cu13/lib",
         "python3", "-c", "import vllm; print('OK:', vllm.__version__)"])
print(r.stdout)
if r.stderr:
    print("STDERR:", r.stderr[-500:])
print("Exit:", r.returncode)
