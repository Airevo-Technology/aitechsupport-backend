"""Download the int8 ONNX bge-m3 model + tokenizer for app/core/local_embed.py.

Usage: python scripts/fetch_local_embed_model.py [target_dir]   (default ~/models/bge-m3)
~570 MB. Source: https://huggingface.co/Xenova/bge-m3. Idempotent (skips files already complete).
"""
from __future__ import annotations

import os
import sys
import urllib.request

BASE = "https://huggingface.co/Xenova/bge-m3/resolve/main"
FILES = {"onnx/model_quantized.onnx": "model_quantized.onnx", "tokenizer.json": "tokenizer.json"}


def main() -> None:
    target = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/models/bge-m3")
    os.makedirs(target, exist_ok=True)
    for remote, local in FILES.items():
        dest = os.path.join(target, local)
        url = f"{BASE}/{remote}"
        size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
        if os.path.isfile(dest) and os.path.getsize(dest) == size:
            print(f"ok   {local} ({size} bytes, already present)")
            continue
        tmp = dest + ".part"
        print(f"get  {local} ({size} bytes) ...")
        urllib.request.urlretrieve(url, tmp)
        if os.path.getsize(tmp) != size:
            raise SystemExit(f"{local}: size mismatch, aborting")
        os.replace(tmp, dest)
    print("done ->", target)


if __name__ == "__main__":
    main()
