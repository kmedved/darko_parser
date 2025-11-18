#!/usr/bin/env python3
"""
NBA CDN PBP & Box Score Cataloguer

This script fetches liveData from the NBA CDN for a list of game IDs.
It analyzes the structure of both playbyplay and boxscore JSON payloads
to identify all unique event "signatures" and data field presence.

Its primary purpose is to detect data drift in the upstream API and
generate configuration files (like `mapping_template.yml`) to help
maintain the nba_scraper parsing logic.

Requirements:
  - requests
  - PyYAML
  - nba_scraper (must be importable via PYTHONPATH)

Usage:
  python scripts/cataloguer.py 0022400001 0022400002 [GAME_ID_3 ...]

Optional arguments:
  -n, --workers   Number of concurrent threads (default: 8)
  -o, --outdir    Directory to write output files (default: out_cdn_catalog)
  -b, --baseline  Baseline cdn_signatures.yml for diffing (default: baseline_signatures.yml)

Note:
  If you need a proxy, configure standard HTTP_PROXY / HTTPS_PROXY
  environment variables; `requests` will use them automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import yaml
except ImportError:
    print("Error: PyYAML not found. Please run: pip install PyYAML")
    sys.exit(1)

try:
    from nba_scraper.mapping.descriptor_norm import canon_str, normalize_descriptor
    from nba_scraper.cdn_parser import _family_from_action
except ImportError:
    print("Error: nba_scraper.mapping.descriptor_norm or cdn_parser not found.")
    sys.exit(1)

CACHE_DIR = Path("cache_raw")
CACHE_PBP_DIR = CACHE_DIR / "cache_pbp"
CACHE_BOX_DIR = CACHE_DIR / "cache_box"
CACHE_PBP_DIR.mkdir(parents=True, exist_ok=True)
CACHE_BOX_DIR.mkdir(parents=True, exist_ok=True)

CDN_PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json"
CDN_BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"

# If True, only existing cache is used; no network requests.
FORCE_CACHE = False


def make_session() -> requests.Session:
    """
    Create a requests Session with basic retry behavior.

    Uses standard HTTP(S)_PROXY environment variables if set.
    """
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s


def fetch_json(url: str, timeout: int = 12) -> Dict[str, Any]:
    """
    Fetch JSON from the given URL with retries.

    Relies on `make_session` and standard environment proxy vars.
    """
    session = make_session()
    resp = session.get(url, timeout=timeout)
    if resp.status_code == 404:
        raise RuntimeError(f"HTTP 404: {url}")
    resp.raise_for_status()
    return resp.json()


def fetch_json_cached(gid: str, data_type: str) -> Dict[str, Any]:
    """
    Fetch JSON from cache if available, otherwise from the CDN.

    data_type: "pbp" or "box".

    Honors FORCE_CACHE:
      - If FORCE_CACHE and cache file is missing -> FileNotFoundError
      - If FORCE_CACHE and cache file is unreadable -> RuntimeError
    """
    if data_type == "pbp":
        path = CACHE_PBP_DIR / f"{gid}.json"
        url = CDN_PBP_URL.format(gid=gid)
    elif data_type == "box":
        path = CACHE_BOX_DIR / f"{gid}.json"
        url = CDN_BOX_URL.format(gid=gid)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to read {data_type} cache for {gid}: {e}. Refetching.")
            if FORCE_CACHE:
                raise RuntimeError(f"Failed to read cache {path} and --force-cache is set.")

    if FORCE_CACHE:
        raise FileNotFoundError(f"Cache file {path} not found and --force-cache is set.")

    data = fetch_json(url)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def family(a: Dict[str, Any]) -> str:
    """
    Return canonical family string for a CDN action.

    Uses the same logic as nba_scraper.cdn_parser._family_from_action
    so that mapping keys align with parser behavior.
    """
    return _family_from_action(a) or "unknown"


def signature(a: Dict[str, Any]) -> Tuple[str, str, str, Tuple[str, ...]]:
    """
    Create a granular signature for a PBP action.

    Signature key:
      (family, subType_norm, descriptor_core, qualifiers_tuple)

    - family: canonical family (e.g. "2pt", "3pt", "freethrow", "turnover")
    - subType_norm: canon_str(subType)
    - descriptor_core: normalize_descriptor(...)[0]
    - qualifiers_tuple: normalized qualifier strings.
    """
    quals = tuple(sorted({canon_str(q) for q in (a.get("qualifiers") or []) if q}))

    descriptor_core, style_flags = normalize_descriptor(a.get("descriptor"))
    a["_style_flags"] = style_flags

    fam = family(a)
    return (
        canon_str(fam),
        canon_str(a.get("subType")),
        descriptor_core,
        quals,
    )


def catalog_game_pbp(gid: str) -> Dict[str, Any]:
    """
    Fetch a single PBP payload and return its catalog:

      {
        "gid": str,
        "signatures": Counter[(family, subType, descriptor_core, qualifiers_tuple)],
        "by_family_schema": {family: Counter[field_name -> count]},
        "examples": {signature: [example_dict1, ... up to 3]},
        "family_counts": Counter[family -> action count],
      }
    """
    print(f"[INFO] Starting PBP {gid}...")
    data = fetch_json_cached(gid, "pbp")

    if "game" not in data or "actions" not in data["game"]:
        raise ValueError(f"'game' or 'actions' key not found in PBP {gid}")

    actions = data["game"]["actions"]
    out = {
        "gid": gid,
        "signatures": Counter(),
        "by_family_schema": defaultdict(Counter),
        "examples": defaultdict(list),
        "family_counts": Counter(),
    }

    for a in actions:
        a_type_norm = canon_str(a.get("actionType"))
        sub_type_norm = canon_str(a.get("subType"))

        if a_type_norm == "period" and not sub_type_norm:
            clk = canon_str(a.get("clock", ""))
            inferred = "end" if clk == "pt00m00.00s" else "start"
            a["subType"] = inferred
            a["_inferredPeriodSubType"] = inferred

        sig = signature(a)
        fam = family(a)
        out["signatures"][sig] += 1
        out["family_counts"][fam] += 1

        for k, v in a.items():
            if k.startswith("_"):
                continue
            if v not in (None, "", [], {}):
                out["by_family_schema"][fam][k] += 1

        if len(out["examples"][sig]) < 3:
            keep = {
                k: a[k]
                for k in (
                    "actionNumber",
                    "orderNumber",
                    "period",
                    "clock",
                    "actionType",
                    "subType",
                    "descriptor",
                    "qualifiers",
                    "description",
                    "personId",
                    "playerName",
                    "teamId",
                    "teamTricode",
                    "shotResult",
                    "pointsTotal",
                    "possession",
                    "scoreHome",
                    "scoreAway",
                    "isFieldGoal",
                    "assistPersonId",
                    "blockPersonId",
                    "stealPersonId",
                    "shotActionNumber",
                    "reboundOffensiveTotal",
                    "reboundDefensiveTotal",
                    "turnoverTotal",
                    "foulPersonalTotal",
                    "foulTechnicalTotal",
                    "shotDistance",
                    "x",
                    "y",
                    "area",
                    "areaDetail",
                    "side",
                    "timeActual",
                    "officialId",
                )
                if k in a and a[k] not in (None, "", [], {})
            }

            keep["gid"] = gid
            if "_style_flags" in a and a["_style_flags"]:
                keep["style_flags"] = a["_style_flags"]
            if "_inferredPeriodSubType" in a:
                keep["inferredPeriodSubType"] = a["_inferredPeriodSubType"]

            m = re.match(r"(\d+)\s+of\s+(\d+)$", canon_str(a.get("subType", "")))
            if m:
                keep["ft_n"], keep["ft_m"] = int(m.group(1)), int(m.group(2))

            out["examples"][sig].append(keep)

    print(f"[SUCCESS] Processed PBP {gid} ({len(actions)} actions)")
    return out


def catalog_game_box(gid: str) -> Dict[str, Any]:
    """
    Fetch a single boxscore payload and catalog team/player schemas.

    Returns:
      {
        "gid": str,
        "player_schema": Counter[field_name -> count],
        "player_stats_schema": Counter[field_name -> count],
        "team_stats_schema": Counter[field_name -> count],
      }
    """
    print(f"[INFO] Starting Box {gid}...")
    data = fetch_json_cached(gid, "box")

    if "game" not in data:
        raise ValueError(f"'game' key not found in Box {gid}")

    game = data["game"]
    out = {
        "gid": gid,
        "player_schema": Counter(),
        "player_stats_schema": Counter(),
        "team_stats_schema": Counter(),
    }

    for side_key in ("homeTeam", "awayTeam"):
        team = game.get(side_key) or {}
        if not team:
            continue

        stats = team.get("statistics") or {}
        for k, v in stats.items():
            if v not in (None, "", [], {}):
                out["team_stats_schema"][k] += 1

        for player in team.get("players", []) or []:
            if not player:
                continue

            for k, v in player.items():
                if k == "statistics":
                    continue
                if v not in (None, "", [], {}):
                    out["player_schema"][k] += 1

            pstats = player.get("statistics") or {}
            for k, v in pstats.items():
                if v not in (None, "", [], {}):
                    out["player_stats_schema"][k] += 1

    print(f"[SUCCESS] Processed Box {gid}")
    return out


def merge_pbp_catalogs(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple PBP catalog outputs into one aggregate structure."""
    merged = {
        "signatures": Counter(),
        "by_family_schema": defaultdict(Counter),
        "examples": defaultdict(list),
        "family_counts": Counter(),
        "game_ids": [],
    }

    for p in parts:
        merged["signatures"].update(p["signatures"])
        merged["family_counts"].update(p["family_counts"])
        merged["game_ids"].append(p["gid"])

        for fam, ctr in p["by_family_schema"].items():
            merged["by_family_schema"][fam].update(ctr)

        for sig, exs in p["examples"].items():
            cur = merged["examples"][sig]
            if len(cur) < 3:
                need = 3 - len(cur)
                cur.extend(exs[:need])

    return merged


def merge_box_catalogs(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple boxscore catalog outputs into one aggregate structure."""
    merged = {
        "player_schema": Counter(),
        "player_stats_schema": Counter(),
        "team_stats_schema": Counter(),
        "game_ids": [p["gid"] for p in parts],
    }
    for p in parts:
        merged["player_schema"].update(p["player_schema"])
        merged["player_stats_schema"].update(p["player_stats_schema"])
        merged["team_stats_schema"].update(p["team_stats_schema"])
    return merged


def dump_mapping_template(merged: Dict[str, Any], outdir: Path) -> None:
    """
    Write mapping_template.yml grouped by canonical family.

    Each entry:
      {
        "actionType": family,
        "signatures": [
           {
             "signature": {subType, descriptor, qualifiers},
             "count": int,
             "map_to_event_name": "",
             "map_to_msg_type": int,
           },
           ...
        ],
      }
    """
    SHOT_FAMS = {"2pt", "3pt"}
    MSG_TYPE_MAP = {
        "freethrow": 3,
        "rebound": 4,
        "turnover": 5,
        "foul": 6,
        "violation": 7,
        "substitution": 8,
        "timeout": 9,
        "jumpball": 10,
        "period": 12,
        "game": 15,
    }

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for (fam, sub_type, desc, quals), cnt in merged["signatures"].most_common():
        suggested = MSG_TYPE_MAP.get(fam, 0)
        if fam in SHOT_FAMS:
            suggested = 0

        grouped[fam].append(
            {
                "signature": {
                    "subType": sub_type,
                    "descriptor": desc,
                    "qualifiers": list(quals),
                },
                "count": cnt,
                "map_to_event_name": "",
                "map_to_msg_type": suggested,
            }
        )

    if "period" in grouped:
        for row in grouped["period"]:
            if row["signature"]["subType"] == "end":
                row["map_to_msg_type"] = 13
            else:
                row["map_to_msg_type"] = 12

    output_list: List[Dict[str, Any]] = []
    for fam in sorted(grouped.keys()):
        output_list.append({"actionType": fam, "signatures": grouped[fam]})

    map_file = outdir / "mapping_template.yml"
    map_file.write_text(
        yaml.dump(output_list, sort_keys=False, default_flow_style=False, width=120),
        encoding="utf-8",
    )
    print(f"Wrote mapping template with {len(merged['signatures'])} signatures to {map_file}")


def dump_vocab(merged: Dict[str, Any], outdir: Path) -> None:
    """
    Write descriptor core, descriptor style, and qualifier vocab YAMLs.

    Files:
      - descriptor_core_vocab.yml
      - descriptor_style_vocab.yml
      - qualifier_vocab.yml
    """
    desc_vocab = Counter()
    qual_vocab = Counter()

    for (_fam, _sub, d_core, q_tuple), cnt in merged["signatures"].items():
        for tok in d_core.split():
            if tok:
                desc_vocab[tok] += cnt
        for tok in q_tuple:
            if tok:
                qual_vocab[tok] += cnt

    style_vocab = Counter()
    for _sig, exs in merged["examples"].items():
        for ex in exs:
            if "style_flags" in ex:
                style_vocab.update(ex["style_flags"])

    (outdir / "descriptor_core_vocab.yml").write_text(
        yaml.dump(desc_vocab.most_common(), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    (outdir / "descriptor_style_vocab.yml").write_text(
        yaml.dump(style_vocab.most_common(), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    (outdir / "qualifier_vocab.yml").write_text(
        yaml.dump(qual_vocab.most_common(), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"Wrote vocab files (descriptor core/style, qualifier) to {outdir}")


def dump_diffs(current_yaml: Path, baseline_yaml: Path, outdir: Path) -> None:
    """
    Compare the current cdn_signatures.yml with a baseline and write:

      - new_signatures.yml
      - changed_frequencies.yml
    """
    try:
        cur_text = current_yaml.read_text(encoding="utf-8")
        cur = yaml.safe_load(cur_text) or []
    except Exception as e:
        print(f"[WARN] Failed to read current signatures file: {e}")
        cur = []

    try:
        base_text = baseline_yaml.read_text(encoding="utf-8")
        base = yaml.safe_load(base_text) or []
    except FileNotFoundError:
        print(f"[INFO] No baseline file found at {baseline_yaml}. All signatures will be 'new'.")
        base = []
    except Exception as e:
        print(f"[WARN] Failed to read baseline file: {e}")
        base = []

    key = lambda r: (
        r["actionType"],
        r["subType"],
        r["descriptor"],
        tuple(sorted(r["qualifiers"])),
    )

    cur_idx = {key(r): r for r in cur}
    base_idx = {key(r): r for r in base}

    new_keys = cur_idx.keys() - base_idx.keys()
    new = [cur_idx[k] for k in new_keys]

    changed = []
    for k in cur_idx.keys() & base_idx.keys():
        if cur_idx[k]["count"] != base_idx[k]["count"]:
            rec = dict(cur_idx[k])
            rec["prev_count"] = base_idx[k]["count"]
            changed.append(rec)

    (outdir / "new_signatures.yml").write_text(
        yaml.dump(new, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    (outdir / "changed_frequencies.yml").write_text(
        yaml.dump(changed, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(new)} new signatures and {len(changed)} changed frequencies to {outdir.resolve()}"
    )


def dump_pbp_outputs(merged: Dict[str, Any], outdir: Path, baseline_file: Path) -> None:
    """
    Write all PBP catalog outputs into outdir:

      - cdn_signatures.yml
      - cdn_signature_examples.jsonl
      - cdn_family_schema.yml
      - mapping_template.yml
      - descriptor_core_vocab.yml
      - descriptor_style_vocab.yml
      - qualifier_vocab.yml
      - diffs/{new_signatures.yml, changed_frequencies.yml}
    """
    print(f"\n--- Writing PBP output to {outdir.resolve()} ---")

    sig_rows: List[Dict[str, Any]] = []
    for (fam, sub_type, desc, quals), cnt in merged["signatures"].most_common():
        sig_rows.append(
            {
                "actionType": fam,
                "subType": sub_type,
                "descriptor": desc,
                "qualifiers": list(quals),
                "count": cnt,
            }
        )

    sig_file = outdir / "cdn_signatures.yml"
    sig_file.write_text(
        yaml.dump(sig_rows, sort_keys=False, default_flow_style=False, width=120),
        encoding="utf-8",
    )
    print(f"Wrote {len(sig_rows)} unique PBP signatures to {sig_file}")

    ex_file = outdir / "cdn_signature_examples.jsonl"
    with ex_file.open("w", encoding="utf-8") as f:
        for (fam, sub_type, desc, quals), exs in sorted(merged["examples"].items()):
            rec = {
                "signature": {
                    "actionType": fam,
                    "subType": sub_type,
                    "descriptor": desc,
                    "qualifiers": list(quals),
                },
                "examples": exs,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote PBP examples for {len(merged['examples'])} signatures to {ex_file}")

    fam_schema: List[Dict[str, Any]] = []
    total_actions = sum(merged["family_counts"].values()) or 1

    for fam, ctr in sorted(merged["by_family_schema"].items()):
        fam_count = merged["family_counts"][fam]
        fam_schema.append(
            {
                "family": fam,
                "count": fam_count,
                "pct_of_total": f"{(fam_count / total_actions * 100):.2f}%",
                "fields_by_presence": dict(
                    sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0]))
                ),
            }
        )

    schema_file = outdir / "cdn_family_schema.yml"
    schema_file.write_text(
        yaml.dump(fam_schema, sort_keys=False, default_flow_style=False, width=120),
        encoding="utf-8",
    )
    print(f"Wrote PBP schema presence for {len(fam_schema)} families to {schema_file}")

    dump_mapping_template(merged, outdir)
    dump_vocab(merged, outdir)

    print("\n--- Checking for PBP signature drift ---")
    diff_dir = outdir / "diffs"
    diff_dir.mkdir(parents=True, exist_ok=True)
    dump_diffs(sig_file, baseline_file, diff_dir)


def dump_box_outputs(merged: Dict[str, Any], outdir: Path) -> None:
    """
    Write boxscore schema analysis into outdir/cdn_boxscore_schema.yml.
    """
    print(f"\n--- Writing Box Score output to {outdir.resolve()} ---")

    total_games = len(merged["game_ids"])

    output_data = {
        "games_analyzed": total_games,
        "team_statistics_schema": dict(
            sorted(merged["team_stats_schema"].items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "player_object_schema": dict(
            sorted(merged["player_schema"].items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "player_statistics_schema": dict(
            sorted(
                merged["player_stats_schema"].items(), key=lambda kv: (-kv[1], kv[0])
            )
        ),
    }

    schema_file = outdir / "cdn_boxscore_schema.yml"
    schema_file.write_text(
        yaml.dump(output_data, sort_keys=False, default_flow_style=False, width=120),
        encoding="utf-8",
    )
    print(f"Wrote Box Score schema analysis to {schema_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NBA CDN Play-by-Play and Box Score Cataloguer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "game_ids",
        metavar="GAME_ID",
        nargs="+",
        help="One or more game IDs to process.",
    )
    parser.add_argument(
        "-n",
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent download threads.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="out_cdn_catalog",
        help="Directory to write output files.",
    )
    parser.add_argument(
        "-b",
        "--baseline",
        type=str,
        default="baseline_signatures.yml",
        help="Path to a baseline 'cdn_signatures.yml' file for diffing.",
    )
    parser.add_argument(
        "--force-cache",
        action="store_true",
        help="Only use local cache; fail if games are not cached.",
    )

    args = parser.parse_args()

    global FORCE_CACHE
    FORCE_CACHE = args.force_cache

    outdir = Path(args.outdir)
    baseline_file = Path(args.baseline)
    workers = args.workers
    gids = sorted(set(args.game_ids))

    print("--- Starting NBA CDN Cataloguer ---")
    print(f"Will cache raw JSON to: {CACHE_DIR.resolve()}")
    print(f"Will write outputs to: {outdir.resolve()}")
    print(f"Will diff PBP signatures against: {baseline_file.resolve()}")
    print(f"Processing {len(gids)} games with {workers} workers...")

    pbp_parts: List[Dict[str, Any]] = []
    box_parts: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures: Dict[Any, Tuple[str, str]] = {}
        for gid in gids:
            futures[ex.submit(catalog_game_pbp, gid)] = (gid, "pbp")
            futures[ex.submit(catalog_game_box, gid)] = (gid, "box")

        for fut in as_completed(futures):
            gid, kind = futures[fut]
            try:
                result = fut.result()
                if kind == "pbp":
                    pbp_parts.append(result)
                else:
                    box_parts.append(result)
            except Exception as e:
                print(f"[WARN] Failed to process {kind.upper()} for {gid}: {e}")

    outdir.mkdir(parents=True, exist_ok=True)

    if pbp_parts:
        merged_pbp = merge_pbp_catalogs(pbp_parts)
        dump_pbp_outputs(merged_pbp, outdir, baseline_file)
        gid_file = outdir / "game_ids_used_pbp.txt"
        gid_file.write_text("\n".join(sorted(merged_pbp["game_ids"])) + "\n")
        print(f"Wrote {len(merged_pbp['game_ids'])} PBP game IDs used to {gid_file}")
    else:
        print("\nNo PBP games were successfully processed. Skipping PBP output.")

    if box_parts:
        merged_box = merge_box_catalogs(box_parts)
        dump_box_outputs(merged_box, outdir)
        gid_file = outdir / "game_ids_used_box.txt"
        gid_file.write_text("\n".join(sorted(merged_box["game_ids"])) + "\n")
        print(f"Wrote {len(merged_box['game_ids'])} Box game IDs used to {gid_file}")
    else:
        print("\nNo Box Score games were successfully processed. Skipping Box output.")

    print("\nCataloguing complete.")


if __name__ == "__main__":
    main()
