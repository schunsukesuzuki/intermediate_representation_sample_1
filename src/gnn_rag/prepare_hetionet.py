"""Single optional Compose job: fetch/validate raw Hetionet then sample small KG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fetch_hetionet import ensure_hetionet_inputs
from .public_data import convert_hetionet


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare compact Hetionet from official data")
    parser.add_argument("--raw-dir", type=Path, default=Path("/app/data/raw"))
    parser.add_argument("--output", type=Path, default=Path("/app/data/hetionet_demo"))
    parser.add_argument("--focus", default="Alzheimer's disease")
    parser.add_argument("--max-nodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    nodes, edges = ensure_hetionet_inputs(args.raw_dir, download=not args.no_download)
    result = convert_hetionet(nodes, edges, args.output, focus=args.focus,
                              max_nodes=args.max_nodes, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
