"""Download the TCGA Pan-Cancer Atlas files used by the project from UCSC Xena.

Files land in data/raw/ and are skipped if already present with the expected size.

    python -m src.download
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

HUB = "https://pancanatlas.xenahubs.net/download/"
FILES = {
    # Batch-corrected RNA-seq (EB++), log2(norm_count + 1), ~11k samples x ~20.5k genes.
    "expression.tsv.gz": ("EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz", 331_000_731),
    # Sample annotations: cancer type, sample type (primary tumour, metastasis, normal).
    "phenotype.tsv.gz": ("TCGA_phenotype_denseDataOnlyDownload.tsv.gz", 61_165),
    # Clinical and survival table from the TCGA Clinical Data Resource (Liu et al., Cell 2018).
    "clinical.tsv": ("Survival_SupplementalTable_S1_20171025_xena_sp", 2_419_504),
}

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch(name: str, remote: str, size: int, attempts: int = 8) -> None:
    target = RAW / name
    if target.exists() and target.stat().st_size == size:
        print(f"ok      {name} (already downloaded)")
        return
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"get     {name} ({size / 1e6:.0f} MB)")
    # A dropped connection resumes from the bytes already on disk instead of starting over.
    for attempt in range(1, attempts + 1):
        done = tmp.stat().st_size if tmp.exists() else 0
        if done >= size:
            break
        req = urllib.request.Request(HUB + remote, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "ab" if done else "wb") as out:
                if done and resp.status != 206:
                    out.truncate(0)
                    done = 0
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    print(f"\r        {done / size:6.1%}", end="", flush=True)
        except OSError as e:
            print(f"\n        connection lost ({e.__class__.__name__}), resuming, attempt {attempt}/{attempts}")
    print()
    if tmp.stat().st_size != size:
        raise RuntimeError(f"{name}: expected {size} bytes, got {tmp.stat().st_size}")
    tmp.replace(target)


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (remote, size) in FILES.items():
        fetch(name, remote, size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
