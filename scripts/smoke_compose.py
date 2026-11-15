"""Stdlib-only Compose API smoke check. Run on the host, not inside Docker."""
from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen


def api_call(base: str, path: str, payload: dict | None = None) -> dict:
    req = Request(
        base.rstrip("/") + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None,
        headers={"Content-Type": "application/json"} if payload else {},
        method="POST" if payload else "GET",
    )
    with urlopen(req, timeout=360 if path == "/answer" else 20) as response:
        return json.load(response)


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke-test the running Compose API")
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--generate", action="store_true", help="Also call Ollama /answer")
    args = p.parse_args()
    health = api_call(args.base, "/health")
    assert health["ok"], "Training artifacts unavailable"
    info = api_call(args.base, "/info")
    node_id = info["start_id"]
    query = {"node_id": node_id, "question": "観測済みの関係を根拠IDつきで説明してください。",
             "hops": 2, "top_k": 3}
    evidence = api_call(args.base, "/evidence", query)
    assert "evidence" in evidence and "prompt" in evidence
    evaluation = api_call(args.base, "/evaluation")
    brief = {method: {f"hits@{k}": round(value["metrics"][f"hits@{k}"], 3)
                       for k in (3, 5, 10)}
             for method, value in evaluation["methods"].items()}
    print(json.dumps({"health": health, "info": info,
                      "evidence_paths": len(evidence["evidence"]),
                      "typed_evidence": len(evidence["catalog"]),
                      "evaluation_query_nodes": evaluation["query_nodes"],
                      "hits_at_k": brief}, ensure_ascii=False, indent=2))
    if args.generate:
        reply = api_call(args.base, "/answer", query)
        print("LLM model:", reply["model"])
        print("Selection status:", reply["selection_status"])
        print("Selected evidence IDs:", reply["selected_evidence_ids"])
        print("Verified answer:\n", reply["answer"])


if __name__ == "__main__":
    main()
