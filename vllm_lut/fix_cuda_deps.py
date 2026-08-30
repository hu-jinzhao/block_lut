#!/usr/bin/env python3
"""Patch vLLM .so files to use CUDA 12.8 runtime instead of 13.0."""
import os, glob, subprocess

VLLM_DIR = "/home/hh/.local/lib/python3.12/site-packages/vllm"
PATCHELF = os.path.expanduser("~/.local/bin/patchelf")

def main():
    # Step 1: Replace libcudart.so.13 -> libcudart.so.12 in NEEDED
    print("=== Step 1: Replace NEEDED entries ===")
    for f in sorted(glob.glob(f"{VLLM_DIR}/*.abi3.so")):
        fname = os.path.basename(f)
        subprocess.run([PATCHELF, "--replace-needed", "libcudart.so.13", "libcudart.so.12", f],
                      capture_output=True)
        print(f"  {fname}: replaced NEEDED libcudart.so.13 -> libcudart.so.12")

    # Step 2: Clear symbol versioning for cudart symbols
    print("\n=== Step 2: Clear symbol versioning ===")
    for f in sorted(glob.glob(f"{VLLM_DIR}/*.abi3.so")):
        fname = os.path.basename(f)
        result = subprocess.run(["objdump", "-T", f], capture_output=True, text=True)
        symbols = []
        for line in result.stdout.split("\n"):
            if "libcudart.so.13" in line and "DF *UND*" in line:
                sym = line.strip().split()[-1]
                symbols.append(sym)

        count = 0
        for sym in symbols:
            r = subprocess.run([PATCHELF, "--clear-symbol-version", sym, f],
                              capture_output=True, text=True)
            if r.returncode == 0:
                count += 1
        print(f"  {fname}: cleared {count}/{len(symbols)} cudart symbol versions")

    # Step 3: Also fix _moe_C_stable_libtorch
    print("\n=== Step 3: Verify ===")
    for f in sorted(glob.glob(f"{VLLM_DIR}/*.abi3.so")):
        fname = os.path.basename(f)
        r = subprocess.run(["objdump", "-p", f], capture_output=True, text=True)
        needed = [l.strip() for l in r.stdout.split("\n") if "NEEDED" in l and "cudart" in l]
        if needed:
            print(f"  {fname}: {needed}")

    print("\nAll patching complete!")

if __name__ == "__main__":
    main()
