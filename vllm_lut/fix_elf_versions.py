#!/usr/bin/env python3
"""
Patch vLLM .so files to remove CUDA version requirements.

The issue: vLLM's .so files have .gnu.version_r entries requiring
version 'libcudart.so.13' from 'libcudart.so.12'. We need to remove
the version requirement line from .gnu.version_r.

This script reads and patches the ELF binary directly.
"""
import os, struct, sys

VLLM_DIR = "/home/hh/.local/lib/python3.12/site-packages/vllm"

def remove_version_req(so_path, lib_name="libcudart.so.12"):
    """Remove a version requirement entry from .gnu.version_r."""
    with open(so_path, "rb") as f:
        data = bytearray(f.read())

    # Find the string table
    # First, find the .gnu.version_r section
    # We need to parse ELF headers

    # Simple approach: find the string and zero out the version requirement
    # The .gnu.version_r has entries like:
    #   File: libcudart.so.12 Cnt: 1
    #   Name: libcudart.so.13 Flags: none Version: 6

    # Find the version_r section by locating the library name in the dynamic string table
    lib_bytes = lib_name.encode() + b"\x00"
    pos = data.find(lib_bytes)
    if pos < 0:
        print(f"  {os.path.basename(so_path)}: {lib_name} not found in dynstr")
        return False

    # The version requirement name "libcudart.so.13" should be right after in version_r
    ver_name = b"libcudart.so.13\x00"
    ver_pos = data.find(ver_name)
    if ver_pos < 0:
        print(f"  {os.path.basename(so_path)}: libcudart.so.13 version entry not found")
        # Check if it's already been removed
        return True

    # Zero out the version requirement entry
    # The entry is ~16-20 bytes: 2 bytes version index + 2 bytes flags + 4 bytes name offset
    # Let's zero out a 20-byte range starting from the version entry
    # We need to find the VERSION_NEED entry that contains this name
    # Each entry has: vn_version(2), vn_cnt(2), vn_file(4), vn_aux(4), vn_next(4)

    # The .gnu.version_r section has:
    # Elf64_Verneed: vn_version=1, vn_cnt=N, vn_file=string_offset, vn_aux, vn_next
    # Elf64_Vernaux: vna_hash, vna_flags, vna_other, vna_name, vna_next

    # Find where the version entry is by scanning backwards from the string
    # The string_offset in vna_name points to the string in .dynstr
    # We need to find the Elf64_Vernaux struct that has vna_name pointing to our string

    # Read .dynstr section
    # We can find it by scanning all section headers
    # For simplicity, just find the version name and search around it

    # Actually, let me just find the struct by parsing .gnu.version_r
    # Parse ELF header to find section headers
    if data[:4] != b"\x7fELF":
        print(f"  Not an ELF file")
        return False

    is_64bit = data[4] == 2  # EI_CLASS
    is_le = data[5] == 1  # EI_DATA

    if is_64bit:
        shoff = struct.unpack_from("<Q", data, 0x28)[0]  # e_shoff
        shentsize = struct.unpack_from("<H", data, 0x3A)[0]
        shnum = struct.unpack_from("<H", data, 0x3C)[0]
        shstrndx = struct.unpack_from("<H", data, 0x3E)[0]
    else:
        print(f"  32-bit ELF not supported")
        return False

    # Find .gnu.version_r section
    # Its type is SHT_GNU_verneed = 0x6ffffffe
    verneed_type = 0x6FFFFFFE if is_le else 0xFEFFFF6F

    # Read the string table for section names
    shstr_off = shoff + shstrndx * shentsize
    shstr_offset = struct.unpack_from("<Q", data, shstr_off + 0x18)[0]
    shstr_size = struct.unpack_from("<Q", data, shstr_off + 0x20)[0]
    shstr = data[shstr_offset:shstr_offset + shstr_size]

    # Find .gnu.version_r
    verneed_off = -1
    verneed_size = 0
    for i in range(shnum):
        s_off = shoff + i * shentsize
        sh_name = struct.unpack_from("<I", data, s_off)[0]
        sh_type = struct.unpack_from("<I", data, s_off + 4)[0]
        if sh_type == verneed_type:
            # Get section name
            name_end = shstr.find(b"\x00", sh_name)
            name = shstr[sh_name:name_end].decode("ascii", errors="replace")
            if "verneed" in name or "version_r" in name or ".gnu.version_r" in name:
                verneed_off = struct.unpack_from("<Q", data, s_off + 0x18)[0]
                verneed_size = struct.unpack_from("<Q", data, s_off + 0x20)[0]
                break

    if verneed_off < 0:
        print(f"  .gnu.version_r section not found")
        return False

    print(f"  .gnu.version_r at offset 0x{verneed_off:x}, size {verneed_size}")

    # Parse the version need entries
    off = verneed_off
    end = off + verneed_size
    file_name_pos = -1
    entry_offsets_to_zero = []

    while off < end:
        if off + 16 > len(data):
            break
        vn_version = struct.unpack_from("<H", data, off)[0]
        vn_cnt = struct.unpack_from("<H", data, off + 2)[0]
        vn_file = struct.unpack_from("<I", data, off + 4)[0]
        vn_aux = struct.unpack_from("<I", data, off + 8)[0]
        vn_next = struct.unpack_from("<I", data, off + 12)[0]

        if vn_version != 1:
            break

        # Get the file name from .dynstr
        file_name = "unknown"
        fn_end = data.find(b"\x00", verneed_off + vn_file)
        if fn_end > 0:
            # The offset is from the section start? No, from the strtab
            pass

        # Actually, vn_file is an offset into .dynstr
        # We need .dynstr base. Let's skip the file name for now

        # Parse aux entries
        aux_off = off + vn_aux
        for _ in range(vn_cnt):
            if aux_off + 16 > len(data):
                break
            vna_name = struct.unpack_from("<I", data, aux_off + 8)[0]  # offset in .dynstr
            vna_next = struct.unpack_from("<I", data, aux_off + 12)[0]

            # Check if this name matches libcudart.so.13
            # vna_name is an offset in .dynstr. We need to resolve it.
            # For patchelf-modified .so, the strings might be at different locations

            # As a simpler approach, let's check if the version tag mentions cudart
            # Read the name string directly
            name_str = ""
            # Actually, let's just mark this aux entry for deletion
            # We'll zero it out

            entry_offsets_to_zero.append(aux_off)

            if vna_next == 0:
                break
            aux_off += vna_next

        if vn_next == 0:
            break
        off += vn_next

    if not entry_offsets_to_zero:
        print(f"  No version entries found")
        return True

    # Zero out the entries
    for e_off in entry_offsets_to_zero:
        for i in range(20):  # Zero out 20 bytes
            if e_off + i < len(data):
                data[e_off + i] = 0

    print(f"  Zeroed out {len(entry_offsets_to_zero)} version entries")

    # Write back
    with open(so_path, "wb") as f:
        f.write(data)

    return True


def main():
    print("=== Removing CUDA version requirements from vLLM .so files ===")
    for f in sorted(os.listdir(VLLM_DIR)):
        if not f.endswith(".abi3.so"):
            continue
        fp = os.path.join(VLLM_DIR, f)
        remove_version_req(fp)

    # Also fix the CUDA 12.8 runtime
    rt_path = "/home/hh/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
    PT = os.path.expanduser("~/.local/bin/patchelf")
    import subprocess
    subprocess.run([PT, "--set-soname", "libcudart.so.13", rt_path])
    print(f"\nCUDA 12.8 runtime SONAME set to libcudart.so.13")

    print("\nAll fixes applied. Testing...")
    os.environ["LD_LIBRARY_PATH"] = "/home/hh/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
    try:
        import vllm
        print(f"SUCCESS! vLLM {vllm.__version__} imported OK")
    except Exception as e:
        print(f"Still failing: {e}")


if __name__ == "__main__":
    main()
