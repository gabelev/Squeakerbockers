"""Search the Freesound API for sounds matching a query.

Outputs a CSV (or stdout) listing id, name, duration, license, and the
two preview MP3 URLs. The preview URLs are downloadable with a plain
API token — no OAuth required.

Setup:
    1. Make a free account at https://freesound.org
    2. Apply for a token: https://freesound.org/apiv2/apply/
    3. export FREESOUND_API_KEY=<your_token>

Usage:
    uv run python scripts/freesound_search.py "shoe squeak" --out shoe.csv
    uv run python scripts/freesound_search.py "basketball court" \\
        --license '"Creative Commons 0"' --limit 200 --out court.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import requests

API = "https://freesound.org/apiv2/search/text/"


def search(query: str, license_filter: str | None, limit: int) -> list[dict]:
    key = os.environ.get("FREESOUND_API_KEY")
    if not key:
        sys.exit(
            "FREESOUND_API_KEY env var not set.\n"
            "Apply at https://freesound.org/apiv2/apply/ then export it."
        )

    headers = {"Authorization": f"Token {key}"}
    params = {
        "query": query,
        "fields": "id,name,duration,license,previews,tags",
        "page_size": min(150, limit),
    }
    if license_filter:
        params["filter"] = f"license:{license_filter}"

    results: list[dict] = []
    url: str | None = API
    while url and len(results) < limit:
        if url == API:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
        else:
            resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results.extend(data.get("results", []))
        url = data.get("next")
    return results[:limit]


def main() -> None:
    p = argparse.ArgumentParser(description="Search Freesound by text query")
    p.add_argument("query", help='e.g. "sneaker squeak"')
    p.add_argument(
        "--license",
        default='"Creative Commons 0"',
        help='license filter, e.g. \'"Creative Commons 0"\' (default) '
        'or \'"Attribution"\'',
    )
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out", default="-", help="output CSV path, or '-' for stdout")
    args = p.parse_args()

    results = search(args.query, args.license, args.limit)

    f = sys.stdout if args.out == "-" else open(args.out, "w", newline="")
    try:
        w = csv.writer(f)
        w.writerow(
            ["id", "name", "duration_sec", "license", "preview_hq_mp3", "preview_lq_mp3"]
        )
        for r in results:
            prev = r.get("previews", {}) or {}
            w.writerow(
                [
                    r["id"],
                    r.get("name", ""),
                    r.get("duration", ""),
                    r.get("license", ""),
                    prev.get("preview-hq-mp3", ""),
                    prev.get("preview-lq-mp3", ""),
                ]
            )
    finally:
        if f is not sys.stdout:
            f.close()

    total_dur = sum(float(r.get("duration") or 0) for r in results)
    print(
        f"Wrote {len(results)} results "
        f"(~{total_dur/60:.1f} min total) to {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
