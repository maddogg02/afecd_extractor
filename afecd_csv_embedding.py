# afecd_csv_embedding.py (robust ref casting + column autodetect)
# ---------------------------------------------------------------
# Builds one embeddings CSV from the CSVs produced by afecd_to_csv.py
#
# IN  = r"C:\Users\krced\AFECD TO CSV\AFECDPDF\out"
# OUT = r"C:\Users\krced\AFECD TO CSV\AFECDPDF\out\embeddings_all.csv"
#
# Usage:
#   python afecd_csv_embedding.py
#   # or with options:
#   python afecd_csv_embedding.py --include-summaries --include-shredouts --include-experience
#   python afecd_csv_embedding.py --plain-text

import argparse
import hashlib
import json
import os
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

DEFAULT_DIR = r"C:\Users\krced\AFECD TO CSV\AFECDPDF\out"
DEFAULT_OUT = os.path.join(DEFAULT_DIR, "embeddings_all.csv")

ORDERED_DIGITS = ["1","3","5","7","9","0","2","4","6","8"]
DIGIT_TO_LABEL = {
    "1": "helper", "2": "apprentice", "3": "apprentice",
    "4": "journeyman", "5": "journeyman",
    "6": "craftsman", "7": "craftsman",
    "8": "superintendent", "9": "superintendent",
    "0": "senior_enlisted_leader",
}
LABEL_TO_DIGIT = {
    "helper": "1", "apprentice": "3", "journeyman": "5",
    "craftsman": "7", "superintendent": "9",
    "senior_enlisted_leader": "0",
}

VERB_MAP = {
    "checks": [1,3,5], "performs": [1,3,5], "operates": [1,3,5], "records": [1,3,5],
    "maintains": [1,3,5], "loads": [1,3,5], "stows": [1,3,5], "inspects": [1,3,5],
    "configures": [3,5], "computes": [3,5,7,9], "determines": [3,5,7,9],
    "analyzes": [5,7,9], "troubleshoots": [3,5,7], "tests": [3,5,7],
    "evaluates": [7,9], "advises": [7,9], "leads": [7,9], "supervises": [7,9],
    "manages": [7,9], "plans": [5,7,9], "coordinates": [5,7,9], "monitors": [3,5,7],
    "directs": [7,9], "attaches": [1,3,5], "connects": [1,3,5], "installs": [1,3,5],
    "repairs": [3,5,7], "aligns": [3,5,7], "calibrates": [3,5,7],
}

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def read_csv_if_exists(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
    return pd.DataFrame()

def parse_sc_list(cell: Optional[str]) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)) or cell == "":
        return []
    return [str(x).strip() for x in str(cell).split(";") if str(x).strip()]

def normalize_digits(digs: List[str]) -> List[str]:
    digs = [str(d).strip() for d in digs if str(d).strip()]
    digs = [d for d in digs if d.isdigit()]
    return [d for d in ORDERED_DIGITS if d in digs]

def labels_for_digits(digs: List[str]) -> List[str]:
    return [DIGIT_TO_LABEL.get(d, "unknown") for d in digs]

def digits_for_labels(labels: List[str]) -> List[str]:
    digs = []
    for lbl in labels:
        d = LABEL_TO_DIGIT.get(str(lbl).strip().lower())
        if d: digs.append(d)
    return normalize_digits(digs)

def infer_from_verbs(sentence: str) -> List[str]:
    s = f" {str(sentence).lower()} "
    hits: Set[str] = set()
    for v, ds in VERB_MAP.items():
        if f" {v} " in s:
            for d in ds:
                hits.add(str(d))
    return normalize_digits(list(hits))

def as_str(x) -> str:
    """Robustly cast any cell to a clean string (keeps things like 2.1, not 2.1 with .0)."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # fix pandas float-y refs like '2.0' -> '2.0' (keep), but '2.0.' -> '2.0'
    # later we’ll accept any string; no type gatekeeping
    return s

def clean_ref(raw: str) -> str:
    """Normalize a ref like 2.1, 2.1., '2.1 '."""
    s = as_str(raw)
    s = s.replace("..", ".").strip()
    if s.endswith(".") and s.count(".") == 1:
        s = s[:-1]
    return s

def get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_inputs(indir: str) -> Dict[str, pd.DataFrame]:
    indir = os.path.normpath(indir)
    dfs = {
        "afsc_docs": read_csv_if_exists(os.path.join(indir, "afsc_docs.csv")),
        "duty_sentences": read_csv_if_exists(os.path.join(indir, "duty_sentences.csv")),
        "duty_skill_map": read_csv_if_exists(os.path.join(indir, "duty_skill_map.csv")),
        "canonical_duties": read_csv_if_exists(os.path.join(indir, "canonical_duties.csv")),
        "summaries": read_csv_if_exists(os.path.join(indir, "afsc_summaries.csv")),
        "shredouts": read_csv_if_exists(os.path.join(indir, "shredouts.csv")),
        "experience": read_csv_if_exists(os.path.join(indir, "afsc_experience_hints.csv")),
    }
    print("Loaded:")
    for k, df in dfs.items():
        print(f"  {k:18s}: {len(df):>6} rows")
    return dfs

def merge_doc_rows(df: pd.DataFrame) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if df.empty: return out

    for doc_id, g in df.groupby("doc_id"):
        cem_code = ""
        afsc_title = "(UNKNOWN TITLE)"
        afsc_codes: Set[str] = set()
        afsc_families: Set[str] = set()
        skill_digits: Set[str] = set()
        skill_labels: Set[str] = set()

        for _, r in g.iterrows():
            c = r.get("cem_code");  cem_code = as_str(c) if as_str(c) and not cem_code else cem_code
            t = r.get("afsc_title"); afsc_title = as_str(t) if as_str(t) and afsc_title == "(UNKNOWN TITLE)" else afsc_title
            afsc_codes.update(parse_sc_list(r.get("afsc_codes")))
            afsc_families.update(parse_sc_list(r.get("afsc_families")))
            skill_digits.update(parse_sc_list(r.get("skill_digits")))
            skill_labels.update(parse_sc_list(r.get("skill_labels")))

        digs = normalize_digits(list(skill_digits)) or digits_for_labels(list(skill_labels))
        lbls = labels_for_digits(digs)
        out[as_str(doc_id)] = {
            "doc_id": as_str(doc_id),
            "cem_code": cem_code or "",
            "afsc_title": afsc_title or "(UNKNOWN TITLE)",
            "afsc_codes": sorted(afsc_codes),
            "afsc_families": sorted(afsc_families),
            "skill_digits": digs,
            "skill_labels": lbls,
        }
    print(f"Merged doc index: {len(out)} docs")
    return out

def skill_map_index(df: pd.DataFrame) -> Dict[Tuple[str,str,int], List[str]]:
    m: Dict[Tuple[str,str,int], Set[str]] = {}
    if df.empty: return {}

    # autodetect columns
    ref_col = get_col(df, ["ref", "section_ref", "section"]) or "ref"
    si_col = get_col(df, ["sent_index", "sentence_index", "idx"]) or "sent_index"
    sd_col = get_col(df, ["skill_digit", "digit", "level_digit"]) or "skill_digit"

    for _, r in df.iterrows():
        doc_id = as_str(r.get("doc_id"))
        ref = clean_ref(r.get(ref_col))
        sd_raw = r.get(sd_col)
        try:
            si = int(float(r.get(si_col))) if pd.notna(r.get(si_col)) else None
        except Exception:
            continue
        if not doc_id or not ref or si is None: 
            continue
        if pd.isna(sd_raw): 
            continue
        try:
            d = str(int(sd_raw))
        except Exception:
            d = as_str(sd_raw)
        if not d.isdigit():
            continue

        key = (doc_id, ref, si)
        m.setdefault(key, set()).add(d)

    out = {k: normalize_digits(list(v)) for k, v in m.items()}
    print(f"Duty→skill mappings: {len(out)} sentences with explicit skills")
    return out

def make_id(doc_id: str, ref: str, sent_index: Optional[int], text: str) -> str:
    h = sha1(f"{doc_id}|{ref}|{sent_index}|{text}")[:8]
    return f"{doc_id}:{ref}:{sent_index if sent_index is not None else 'S'}:{h}"

def build_duty_rows(df_duty: pd.DataFrame, doc_idx: Dict[str,dict],
                    skill_idx: Dict[Tuple[str,str,int], List[str]],
                    plain_text: bool) -> List[dict]:
    if df_duty.empty: return []

    # autodetect cols
    ref_col = get_col(df_duty, ["ref", "section_ref", "section"]) or "ref"
    si_col = get_col(df_duty, ["sent_index", "sentence_index", "idx"]) or "sent_index"
    text_col = get_col(df_duty, ["sentence_text", "text"]) or "sentence_text"
    can_col = get_col(df_duty, ["canonical_duty_id", "canonical_id"]) or "canonical_duty_id"

    out = []
    for _, r in df_duty.iterrows():
        doc_id = as_str(r.get("doc_id"))
        ref = clean_ref(r.get(ref_col))
        sentence = as_str(r.get(text_col))
        can_id = as_str(r.get(can_col))

        try:
            si_raw = r.get(si_col)
            si = int(float(si_raw)) if pd.notna(si_raw) else None
        except Exception:
            si = None

        if not doc_id or not ref or not sentence:
            continue

        doc = doc_idx.get(doc_id, {
            "cem_code": "",
            "afsc_title": "(UNKNOWN TITLE)",
            "afsc_codes": [],
            "afsc_families": [],
            "skill_digits": [],
            "skill_labels": [],
        })

        digs = skill_idx.get((doc_id, ref, si), [])
        if not digs:
            digs = infer_from_verbs(sentence)
        if not digs:
            digs = doc.get("skill_digits", [])
        if not digs:
            digs = ["1","3","5"]
        lbls = labels_for_digits(digs)

        text = sentence if plain_text else f"[{doc.get('afsc_title')} | Duty {ref} | Skills: {','.join(digs)}] {sentence}"
        rid = make_id(doc_id, ref, si, sentence)

        meta = {
            "doc_id": doc_id,
            "cem_code": doc.get("cem_code", ""),
            "afsc_title": doc.get("afsc_title", "(UNKNOWN TITLE)"),
            "afsc_codes": doc.get("afsc_codes", []),
            "afsc_families": doc.get("afsc_families", []),
            "ref": ref,
            "sent_index": si,
            "skill_digits": digs,
            "skill_labels": lbls,
            "canonical_duty_id": can_id,
            "source_type": "duty",
        }
        out.append({
            "id": rid, "doc_id": doc_id, "ref": ref, "sent_index": si,
            "text": text, "metadata_json": json.dumps(meta, ensure_ascii=False),
        })

    print(f"Duty rows: {len(out)}")
    return out

def build_summary_rows(df_sum: pd.DataFrame, doc_idx: Dict[str,dict], plain_text: bool) -> List[dict]:
    if df_sum.empty: return []
    text_col = get_col(df_sum, ["text", "summary"]) or "text"
    out = []
    for _, r in df_sum.iterrows():
        doc_id = as_str(r.get("doc_id"))
        txt = as_str(r.get(text_col))
        if not doc_id or not txt: continue
        doc = doc_idx.get(doc_id, {})
        digs = doc.get("skill_digits", [])
        lbls = labels_for_digits(digs)
        text = txt if plain_text else f"[{doc.get('afsc_title','(UNKNOWN TITLE)')} | Summary | Skills: {','.join(digs)}] {txt}"
        rid = make_id(doc_id, "summary", None, txt)
        meta = {
            "doc_id": doc_id,
            "cem_code": doc.get("cem_code", ""),
            "afsc_title": doc.get("afsc_title", "(UNKNOWN TITLE)"),
            "afsc_codes": doc.get("afsc_codes", []),
            "afsc_families": doc.get("afsc_families", []),
            "ref": "summary",
            "sent_index": None,
            "skill_digits": digs,
            "skill_labels": lbls,
            "canonical_duty_id": "",
            "source_type": "summary",
        }
        out.append({
            "id": rid, "doc_id": doc_id, "ref": "summary", "sent_index": None,
            "text": text, "metadata_json": json.dumps(meta, ensure_ascii=False),
        })
    print(f"Summary rows: {len(out)}")
    return out

def build_shredout_rows(df_sh: pd.DataFrame, doc_idx: Dict[str,dict], plain_text: bool) -> List[dict]:
    if df_sh.empty: return []
    prim_col = get_col(df_sh, ["primary_aircraft", "text"]) or "primary_aircraft"
    out = []
    for _, r in df_sh.iterrows():
        doc_id = as_str(r.get("doc_id"))
        prim = as_str(r.get(prim_col))
        suf = as_str(r.get("suffix"))
        if not doc_id or not prim: continue
        doc = doc_idx.get(doc_id, {})
        head = f"{doc.get('afsc_title','(UNKNOWN TITLE)')} | Shredout {suf}" if suf else f"{doc.get('afsc_title','(UNKNOWN TITLE)')} | Shredout"
        text = prim if plain_text else f"[{head}] {prim}"
        rid = make_id(doc_id, "shredout", None, prim)
        meta = {
            "doc_id": doc_id,
            "cem_code": doc.get("cem_code", ""),
            "afsc_title": doc.get("afsc_title", "(UNKNOWN TITLE)"),
            "afsc_codes": doc.get("afsc_codes", []),
            "afsc_families": doc.get("afsc_families", []),
            "ref": "shredout",
            "sent_index": None,
            "skill_digits": [],
            "skill_labels": [],
            "canonical_duty_id": "",
            "source_type": "shredout",
        }
        out.append({
            "id": rid, "doc_id": doc_id, "ref": "shredout", "sent_index": None,
            "text": text, "metadata_json": json.dumps(meta, ensure_ascii=False),
        })
    print(f"Shredout rows: {len(out)}")
    return out

def build_experience_rows(df_exp: pd.DataFrame, doc_idx: Dict[str,dict], plain_text: bool) -> List[dict]:
    if df_exp.empty: return []
    raw_col = get_col(df_exp, ["raw_text", "text"]) or "raw_text"
    ref_col = get_col(df_exp, ["section_ref", "ref", "section"]) or "section_ref"
    out = []
    for _, r in df_exp.iterrows():
        doc_id = as_str(r.get("doc_id"))
        raw = as_str(r.get(raw_col))
        ref = clean_ref(r.get(ref_col)) or "3.4"
        target = as_str(r.get("target_skill_digit"))
        if not doc_id or not raw: continue
        doc = doc_idx.get(doc_id, {})
        digs = normalize_digits([target]) if target.isdigit() else []
        lbls = labels_for_digits(digs)
        text = raw if plain_text else f"[{doc.get('afsc_title','(UNKNOWN TITLE)')} | Experience | Skills: {','.join(digs) if digs else '-'}] {raw}"
        rid = make_id(doc_id, ref, None, raw)
        meta = {
            "doc_id": doc_id,
            "cem_code": doc.get("cem_code", ""),
            "afsc_title": doc.get("afsc_title", "(UNKNOWN TITLE)"),
            "afsc_codes": doc.get("afsc_codes", []),
            "afsc_families": doc.get("afsc_families", []),
            "ref": ref,
            "sent_index": None,
            "skill_digits": digs,
            "skill_labels": lbls,
            "canonical_duty_id": "",
            "source_type": "experience",
        }
        out.append({
            "id": rid, "doc_id": doc_id, "ref": ref, "sent_index": None,
            "text": text, "metadata_json": json.dumps(meta, ensure_ascii=False),
        })
    print(f"Experience rows: {len(out)}")
    return out

def run(indir: str, out_csv: str, include_summaries: bool, include_shredouts: bool, include_experience: bool, plain_text: bool):
    dfs = load_inputs(indir)
    if dfs["afsc_docs"].empty or dfs["duty_sentences"].empty:
        raise SystemExit("ERROR: afsc_docs.csv or duty_sentences.csv is missing/empty. Re-run the parser first.")

    doc_idx = merge_doc_rows(dfs["afsc_docs"])
    skill_idx = skill_map_index(dfs["duty_skill_map"])

    rows: List[dict] = []
    rows += build_duty_rows(dfs["duty_sentences"], doc_idx, skill_idx, plain_text)
    if include_summaries:  rows += build_summary_rows(dfs["summaries"], doc_idx, plain_text)
    if include_shredouts:  rows += build_shredout_rows(dfs["shredouts"], doc_idx, plain_text)
    if include_experience: rows += build_experience_rows(dfs["experience"], doc_idx, plain_text)

    if not rows:
        raise SystemExit("Nothing to write — no rows assembled (check inputs/flags).")

    out_df = pd.DataFrame(rows, columns=["id","doc_id","ref","sent_index","text","metadata_json"]).drop_duplicates(subset=["id"])
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nWrote embeddings CSV: {os.path.abspath(out_csv)}  (rows={len(out_df)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default=DEFAULT_DIR)
    ap.add_argument("--out", dest="out_csv", default=DEFAULT_OUT)
    ap.add_argument("--include-summaries", action="store_true")
    ap.add_argument("--include-shredouts", action="store_true")
    ap.add_argument("--include-experience", action="store_true")
    ap.add_argument("--plain-text", action="store_true")
    args = ap.parse_args()

    run(
        indir=args.indir,
        out_csv=args.out_csv,
        include_summaries=args.include_summaries,
        include_shredouts=args.include_shredouts,
        include_experience=args.include_experience,
        plain_text=args.plain_text,
    )
