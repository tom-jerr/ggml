#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import mmap
import os
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

# gguf_type (from gguf.h)
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

GGUF_TYPE_NAME = {
    0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32",
    6: "f32", 7: "bool", 8: "string", 9: "array", 10: "u64", 11: "i64", 12: "f64",
}

# Minimal ggml_type names (enough for MNIST f32; others show as type_id)
GGML_TYPE_NAME = {
    0: "F32",
    1: "F16",
    16: "BF16",
    # Quantized types vary across versions; keep numeric if unknown.
}

# Byte size for a few common ggml types (for non-quantized)
GGML_TYPE_ELEM_SIZE = {
    0: 4,   # F32
    1: 2,   # F16
    16: 2,  # BF16
}

def align_up(x: int, a: int) -> int:
    r = x % a
    return x if r == 0 else x + (a - r)

@dataclass
class TensorInfo:
    name: str
    n_dims: int
    dims: List[int]
    ggml_type: int
    rel_offset: int  # relative to tensor_data blob start

    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= int(d)
        return n

    def byte_size_if_known(self) -> Optional[int]:
        # Only accurate for non-quantized types we know sizes for.
        s = GGML_TYPE_ELEM_SIZE.get(self.ggml_type)
        if s is None:
            return None
        return self.n_elements() * s

class Reader:
    def __init__(self, mm: mmap.mmap):
        self.mm = mm
        self.off = 0

    def read(self, n: int) -> bytes:
        b = self.mm[self.off:self.off+n]
        if len(b) != n:
            raise EOFError("Unexpected EOF")
        self.off += n
        return b

    def u8(self) -> int:  return struct.unpack("<B", self.read(1))[0]
    def i8(self) -> int:  return struct.unpack("<b", self.read(1))[0]
    def u16(self) -> int: return struct.unpack("<H", self.read(2))[0]
    def i16(self) -> int: return struct.unpack("<h", self.read(2))[0]
    def u32(self) -> int: return struct.unpack("<I", self.read(4))[0]
    def i32(self) -> int: return struct.unpack("<i", self.read(4))[0]
    def u64(self) -> int: return struct.unpack("<Q", self.read(8))[0]
    def i64(self) -> int: return struct.unpack("<q", self.read(8))[0]
    def f32(self) -> float: return struct.unpack("<f", self.read(4))[0]
    def f64(self) -> float: return struct.unpack("<d", self.read(8))[0]

    def gguf_string(self) -> str:
        # GGUF string: uint64 length + bytes (no '\0') :contentReference[oaicite:1]{index=1}
        n = self.u64()
        raw = self.read(int(n))
        return raw.decode("utf-8", errors="replace")

def parse_value(r: Reader, vtype: int) -> Any:
    if vtype == GGUF_TYPE_UINT8:   return r.u8()
    if vtype == GGUF_TYPE_INT8:    return r.i8()
    if vtype == GGUF_TYPE_UINT16:  return r.u16()
    if vtype == GGUF_TYPE_INT16:   return r.i16()
    if vtype == GGUF_TYPE_UINT32:  return r.u32()
    if vtype == GGUF_TYPE_INT32:   return r.i32()
    if vtype == GGUF_TYPE_UINT64:  return r.u64()
    if vtype == GGUF_TYPE_INT64:   return r.i64()
    if vtype == GGUF_TYPE_FLOAT32: return r.f32()
    if vtype == GGUF_TYPE_FLOAT64: return r.f64()
    if vtype == GGUF_TYPE_BOOL:    return (r.i8() != 0)  # stored as int8 :contentReference[oaicite:2]{index=2}
    if vtype == GGUF_TYPE_STRING:  return r.gguf_string()
    if vtype == GGUF_TYPE_ARRAY:
        elem_type = r.i32()
        n = r.u64()
        arr = []
        for _ in range(int(n)):
            arr.append(parse_value(r, elem_type))
        return {"elem_type": elem_type, "values": arr}
    raise ValueError(f"Unknown gguf value type: {vtype}")

def parse_gguf(path: str) -> Tuple[Dict[str, Any], List[TensorInfo], int, int, int, int]:
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        r = Reader(mm)

        magic = r.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Bad magic: {magic!r} (expected b'GGUF')")

        version = r.u32()
        tensor_count = r.u64()
        kv_count = r.u64()

        meta: Dict[str, Any] = {}
        alignment = 32  # default :contentReference[oaicite:3]{index=3}

        for _ in range(int(kv_count)):
            key = r.gguf_string()
            vtype = r.i32()
            val = parse_value(r, vtype)
            meta[key] = {"type": vtype, "value": val}

            # capture alignment if present
            if key == "general.alignment":
                # spec says uint32 :contentReference[oaicite:4]{index=4}
                if vtype == GGUF_TYPE_UINT32:
                    alignment = int(val)

        tensors: List[TensorInfo] = []
        for _ in range(int(tensor_count)):
            name = r.gguf_string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(int(n_dims))]
            ggml_type = r.i32()   # ggml_type enum
            rel_off  = r.u64()    # relative to tensor_data blob :contentReference[oaicite:5]{index=5}
            tensors.append(TensorInfo(name=name, n_dims=int(n_dims),
                                      dims=[int(x) for x in dims],
                                      ggml_type=int(ggml_type),
                                      rel_offset=int(rel_off)))

        # tensor_data blob starts after padding to alignment
        data_offset = align_up(r.off, alignment)

        mm.close()

    return meta, tensors, version, int(tensor_count), int(kv_count), data_offset

def fmt_type(t: int, mapping: Dict[int, str]) -> str:
    return mapping.get(t, str(t))

def main():
    ap = argparse.ArgumentParser(description="Inspect GGUF file structure (header, metadata, tensors).")
    ap.add_argument("path", help="Path to .gguf file (e.g., mnist-cnn-f32.gguf)")
    ap.add_argument("--no-meta", action="store_true", help="Do not print metadata")
    ap.add_argument("--no-tensors", action="store_true", help="Do not print tensors list")
    ap.add_argument("--grep", default=None, help="Only show metadata keys containing this substring")
    args = ap.parse_args()

    meta, tensors, version, n_tensors, n_kv, data_offset = parse_gguf(args.path)
    fsize = os.path.getsize(args.path)

    print(f"File: {args.path}")
    print(f"Size: {fsize} bytes")
    print(f"GGUF version: {version}")
    print(f"KV count: {n_kv}")
    print(f"Tensor count: {n_tensors}")
    # alignment comes from metadata or default; compute from data_offset padding behavior:
    # (we don't store it explicitly, but can infer if general.alignment exists)
    align_val = meta.get("general.alignment", None)
    align_show = align_val["value"] if (align_val and align_val["type"] == GGUF_TYPE_UINT32) else 32
    print(f"Alignment: {align_show}")
    print(f"Tensor data blob offset (file): {data_offset}")
    print("")

    if not args.no_meta:
        print("=== Metadata (KV) ===")
        keys = sorted(meta.keys())
        for k in keys:
            if args.grep and args.grep not in k:
                continue
            t = meta[k]["type"]
            v = meta[k]["value"]
            tname = fmt_type(t, GGUF_TYPE_NAME)
            if isinstance(v, dict) and "values" in v and "elem_type" in v:
                et = v["elem_type"]
                etname = fmt_type(et, GGUF_TYPE_NAME)
                print(f"{k} : array[{etname}] len={len(v['values'])}")
            else:
                print(f"{k} : {tname} = {v}")
        print("")

    if not args.no_tensors:
        print("=== Tensors ===")
        for i, ti in enumerate(tensors):
            tname = fmt_type(ti.ggml_type, GGML_TYPE_NAME)
            abs_off = data_offset + ti.rel_offset
            bsz = ti.byte_size_if_known()
            bsz_str = f"{bsz} bytes" if bsz is not None else "unknown-size(quantized or unsupported type table)"
            print(f"[{i:4d}] {ti.name}")
            print(f"       dims={ti.dims}  n_elems={ti.n_elements()}  type={tname}({ti.ggml_type})")
            print(f"       rel_off={ti.rel_offset}  abs_off={abs_off}  size≈{bsz_str}")
        print("")

if __name__ == "__main__":
    main()
