"""Cached client for the TypeSafe System One API (Jev).

Every request/response pair is stored in data/cache/<sha256>.json so that
experiments replay offline. `rep` distinguishes deliberate repeated samples
of an identical request.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import requests

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
KEY_FILE = Path.home() / ".config" / "typesafe" / "key"


def _key():
    return os.environ.get("TYPESAFE_API_KEY") or KEY_FILE.read_text().strip()


def noul(instructions):
    return {"type": "noul", "instructions": instructions}


def choice(instructions, criteria):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions, levels):
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def ask(state, questions, rep=0, offline=False):
    body = {"model": MODEL, "state": state, "questions": questions}
    h = hashlib.sha256(json.dumps([body, rep], sort_keys=True).encode()).hexdigest()
    path = CACHE / f"{h}.json"
    if path.exists():
        return json.loads(path.read_text())["response"]
    if offline:
        raise KeyError(f"not cached: {h}")
    delay = 1.0
    for _ in range(8):
        t0 = time.time()
        r = requests.post(URL, json=body, timeout=60,
                          headers={"Authorization": f"Bearer {_key()}"})
        if r.status_code in (429, 529, 502, 503):
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
        resp = r.json()
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"request": body, "rep": rep, "response": resp,
                                    "latency_s": time.time() - t0,
                                    "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
        return resp
    raise RuntimeError(f"gave up after retries: {r.status_code} {r.text[:200]}")
