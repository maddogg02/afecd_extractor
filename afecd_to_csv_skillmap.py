# afecd_to_csv.py (fixed)
# -----------------------
# Parse AFECD PDF -> clean CSVs (docs, summaries, duties, skill map, canonical duties, embeddings, shredouts, experience)
#
# Fixes vs last version:
# - Skip front-matter (SECTION I, DAFECD, SUMMARY OF REVISIONS, etc.)
# - Skill digit now taken from the **4th character** of AFSC code (canonical AFSC convention)
# - Collapses soft line breaks so single sentences don’t get split (e.g., “through-flight, and post flight…”)
# - Filters out bare “2.x.” heading lines (no longer treated as sentences)
# - Tighter shredouts parser + plausibility filter (ignores “is applicable to…” notes)
# - Safer doc segmentation (no “look-back” that contaminated blocks)
#
# Usage:
#   pip install pdfplumber pandas rapidfuzz
#   python afecd_to_csv.py --pdf "C:\Users\krced\AFECD TO CSV\AFECDPDF\Air Force Enlisted Classification Directory AFECD.pdf" --out ./afecd_out --include-experience

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import pdfplumber
from rapidfuzz.utils import default_process

# -----------------------------
# Normalization helpers
# -----------------------------

REPLACEMENTS = {
    "\uf0b7": "-",
    "\uf0a7": "-",
    "": " ",           # star bullet
    "\u00ad": "",       # soft hyphen
    "–": "-", "—": "-",
}

IGNORED_TITLES = {
    "DEPARTMENT OF THE AIR FORCE",
    "SUMMARY OF REVISIONS",
    "DAFECD",
    "SECTION I",
    "SECTION I-A",
    "SECTION II",
    "AUTHORIZED PREFIXES",
    "AIR FORCE SPECIALTY CODES",
    "AIR FORCE REPORTING IDENTIFIERS",
    "SPACE FORCE REPORTING IDENTIFIERS",
}

# Headers
RE_SUMMARY = re.compile(r"^\s*1\.\s*(Specialty\s+Summary\.?)", re.I | re.M)
RE_DUTIES  = re.compile(r"^\s*2\.\s*(Duties\s+and\s+Responsibilities:?)", re.I | re.M)
RE_QUALS   = re.compile(r"^\s*3\.\s*(Specialty\s+Qualifications:?)", re.I | re.M)
RE_SHRED   = re.compile(r"^\s*4\.\s*\*?Specialty\s+Shredouts:?", re.I | re.M)

# Titles (big all-caps lines)
RE_STAR_TITLE = re.compile(r"^\s*([A-Z][A-Z0-9 \-/&()’,']{6,})\s*(?:\([^)]*\))?\s*$")

# AFSC ladder patterns
RE_CEM = re.compile(r"\bCEM\s+Code\s+([0-9][A-Z0-9][0-9][0-9][0-9](?:[A-Z\*]+)?)", re.I)
RE_AFSC_LADDER_PAIR = re.compile(
    r"\bAFSC\s+([0-9][A-Z0-9][0-9][0-9][0-9](?:[A-Z\*]+)?)\s*,\s*([A-Za-z][A-Za-z \-\/&]+?)(?=(?:\s{2,}|\s*\bAFSC\b|$))",
    re.S,
)

# Experience lines (3.4.x.)
RE_EXPERIENCE_LINE = re.compile(
    r"^\s*3\.4\.\d+\.\s+([0-9][A-Z0-9][0-9][0-9][0-9](?:[A-Z\*]+)?)\.\s+(.*)$", re.M
)

# Replace the old RE_SHRED_PAIR with this more permissive one
# – works across lines, allows aircraft codes that start with uppercase
RE_SHRED_PAIR = re.compile(
    r"(?mx)"                              # multiline, verbose
    r"(?:^|\s)"                           # start of line or whitespace
    r"([A-Z])"                            # 1) single-letter suffix
    r"(?:\s+|[^\S\r\n]+)"                 # spaces/tabs
    r"(.+?)"                              # 2) description / primary aircraft (lazy)
    r"(?=(" 
    r"(?:\s+[A-Z]\s+)"                    # lookahead to next suffix letter
    r"|$"                                 # or end of block
    r"))"
)

# Sentence split (after .,?,! followed by uppercase or bracket)
SENT_SPLIT = re.compile(
    r"(?<!\b[A-Z])(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<=[\.\?!])\s+(?=[A-Z(\[])"
)

DIGIT_LABEL = {
    "1": "helper",
    "2": "apprentice",
    "3": "apprentice",
    "4": "journeyman",
    "5": "journeyman",
    "6": "craftsman",
    "7": "craftsman",
    "8": "superintendent",
    "9": "superintendent",
    "0": "senior_enlisted_leader",
}

TITLE_TO_LABEL_HINTS = {
    "senior enlisted leader": "senior_enlisted_leader",
    "superintendent": "superintendent",
    "craftsman": "craftsman",
    "journeyman": "journeyman",
    "apprentice": "apprentice",
    "helper": "helper",
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

AFSC_CODE_ANY = re.compile(r"\b[0-9][A-Z0-9][0-9][0-9][0-9](?:[A-Z\*]+)?\b")

def clean_text(s: str) -> str:
    if not s: return s
    for k, v in REPLACEMENTS.items():
        s = s.replace(k, v)
    # de-hyphenate: "through- flight" -> "through-flight"
    s = re.sub(r"-\s+", "-", s)
    # collapse double spaces and strip
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    return s.strip()

def collapse_soft_linebreaks(s: str) -> str:
    """
    Merge linebreaks inside paragraphs while preserving true section/list breaks.
    Rule of thumb: if a newline is **not** preceded by .?! and the next token
    starts lowercase/number/paren/‘and/or/to’, replace with space.
    Also keep numbered headers (2.x., 3.x.)
    """
    # protect numbered headers with markers
    s = re.sub(r"(?m)^(2\.\d+\.)", r"@@HDR@@\1", s)
    s = re.sub(r"(?m)^(3\.\d+\.)", r"@@HDR@@\1", s)

    # merge “soft” newlines
    s = re.sub(r"(?<![.!?])\n(?!\s*@@HDR@@)", " ", s)

    # restore headers
    s = s.replace("@@HDR@@", "")
    return s

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "untitled"

def afsc_family(code: str) -> str:
    return code[:3] if len(code) >= 3 else code

def ladder_digit_from_code(code: str) -> Optional[str]:
    """
    Correct AFSC rule: the **4th character** (index 3) is the skill level digit (1/3/5/7/9).
    Works even with trailing letters or '*', e.g. 1A133* -> '3'
    """
    code = code.strip()
    if len(code) >= 4 and code[3].isdigit():
        return code[3]
    return None

def level_label_from_title(title: str) -> Optional[str]:
    t = title.strip().lower()
    for k, v in TITLE_TO_LABEL_HINTS.items():
        if k in t:
            return v
    return None

def canonical_duty_id(text: str) -> str:
    t = default_process(text.lower())
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    return sha1(t)[:40]

def sentence_split(paragraph: str) -> List[str]:
    parts = re.split(SENT_SPLIT, paragraph.strip())
    return [p.strip().strip('"').strip() for p in parts if p.strip()]

def infer_skills_for_sentence(sentence: str) -> Tuple[List[int], str, str]:
    s = sentence.lower()
    hits: Set[int] = set()
    matched_verbs: List[str] = []
    for v, digs in VERB_MAP.items():
        if re.search(rf"\b{re.escape(v)}\b", s):
            hits.update(digs); matched_verbs.append(v)
    if not hits:
        return [1,3,5], "", "low"
    conf = "high" if len(matched_verbs) >= 3 else "medium"
    return sorted(hits), "verbs=" + "|".join(sorted(set(matched_verbs))), conf

@dataclass
class AFSCEntry:
    cem_code: Optional[str] = None
    title: Optional[str] = None
    summary: str = ""
    duties: str = ""
    quals: str = ""
    ladders: List[Tuple[str, str]] = field(default_factory=list)
    shred_block: str = ""

def extract_pages(pdf_path: str) -> List[str]:
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            txt = clean_text(p.extract_text() or "")
            out.append(txt)
    return out

def find_title_lines(lines: List[str]) -> List[int]:
    idxs = []
    for i, ln in enumerate(lines):
        m = RE_STAR_TITLE.match(ln)
        if not m: continue
        title = m.group(1).strip()
        # ignore obvious front-matter headings
        if title in IGNORED_TITLES: continue
        # must look “AFSC-like” in neighborhood OR have Summary/Duties later
        window = "\n".join(lines[i:i+40])
        if not (RE_SUMMARY.search(window) or "AFSC " in window or "CEM Code" in window or RE_DUTIES.search(window)):
            continue
        idxs.append(i)
    return idxs

def cut_docs(big_text: str) -> List[str]:
    lines = big_text.splitlines()
    title_idxs = find_title_lines(lines)
    if not title_idxs:
        return [big_text]

    title_idxs.append(len(lines))
    blocks = []
    for a, b in zip(title_idxs, title_idxs[1:]):
        # start at the title line exactly (no look-back to avoid contamination)
        seg = "\n".join(lines[a:b]).strip()
        blocks.append(seg)
    return blocks

def parse_ladders(text: str) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    cem = None
    mcem = RE_CEM.search(text)
    if mcem: cem = mcem.group(1)

    ladders: List[Tuple[str, str]] = []
    for m in RE_AFSC_LADDER_PAIR.finditer(text):
        ladders.append((m.group(1).strip(), m.group(2).strip().strip(".")))

    # dedupe
    seen = set(); out = []
    for code, lvl in ladders:
        key = (code, lvl.lower())
        if key not in seen:
            seen.add(key); out.append((code, lvl))
    return cem, out

def extract_section(block: str, start_re: re.Pattern, end_res: List[re.Pattern]) -> str:
    mstart = start_re.search(block)
    if not mstart: return ""
    start = mstart.end()
    end = len(block)
    for er in end_res:
        mend = er.search(block, start)
        if mend:
            end = min(end, mend.start())
    txt = block[start:end].strip()
    # collapse soft breaks inside sections
    return collapse_soft_linebreaks(txt)

def title_from_block(block: str) -> str:
    for ln in block.splitlines():
        m = RE_STAR_TITLE.match(ln)
        if not m: continue
        t = re.sub(r"\s*\(.*?\)\s*$", "", m.group(1).strip())
        if t not in IGNORED_TITLES:
            return t
    # fallback—use noun phrase after "determine the ..."
    summ = extract_section(block, RE_SUMMARY, [RE_DUTIES, RE_QUALS, RE_SHRED])
    m = re.search(r"determine\s+the\s+([A-Za-z0-9 \-\/&]+?)\s+\(", summ)
    return (m.group(1).strip().upper() if m else "(UNKNOWN TITLE)")

def unify_ladder(ladders: List[Tuple[str, str]]) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (codes, digits_sorted, labels_sorted)
    """
    codes = [c for c, _ in ladders]
    labels_by_digit: Dict[str, str] = {}
    digits: Set[str] = set()
    for code, level in ladders:
        d = ladder_digit_from_code(code)
        if not d: continue
        digits.add(d)
        labels_by_digit[d] = level_label_from_title(level) or DIGIT_LABEL.get(d, "unknown")
    order = ["1","3","5","7","9","0","2","4","6","8"]
    digits_sorted = [d for d in order if d in digits]
    labels_sorted = [labels_by_digit[d] for d in digits_sorted]
    return codes, digits_sorted, labels_sorted

def parse_shredouts(block: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    m = RE_SHRED.search(block)
    if not m: return out
    start = m.end()
    # stop at next major header or end
    end = len(block)
    for pat in [RE_SUMMARY, RE_DUTIES, RE_QUALS, RE_SHRED]:
        mm = pat.search(block, start)
        if mm: end = min(end, mm.start())
    payload = clean_text(block[start:end])

    def plausible(text: str) -> bool:
        t = text.strip()
        tl = t.lower()
        if not t: return False
        if tl.startswith(("is applicable", "are applicable", "note:", "shredout", "cem code", "afsc ")):
            return False
        if "DAFECD" in t or "Program Management" in t:
            return False
        # aircraft / crew cues
        cues = (" C-", " KC-", " EC-", " MC-", " WC-", " HC-", " HH-",
                " E-", " F-", " B-", " RQ-", " MQ-", " RC-", " LC-", " UH-",
                "Loadmaster", "Flight Engineer", "Boom Operator", "Sensor Operator",
                "Mission Operator", "Radar", "Systems Operator")
        return any(c in t for c in cues)

    for m in RE_SHRED_PAIR.finditer(payload):
        suf = m.group(1).strip()
        txt = m.group(2).strip()
        # trim at known breakers
        txt = re.split(r"\b(?:NOTE:|DAFECD,|\bCEM\s+Code\b|\bAFSC\s+)", txt)[0].strip()
        if plausible(txt):
            out.append((suf, txt))
    return out

def duties_to_sentences(duties_text: str) -> Tuple[List[str], Dict[int, str]]:
    """
    Collapse soft breaks, split by headers 2.x., then sentence split.
    Returns (sentences, sent_index -> ref)
    """
    if not duties_text: return [], {}
    text = collapse_soft_linebreaks(duties_text)

    # Tokenize by "2.x." headers while keeping the header as a ref
    parts = re.split(r"(?=(?:^|\s)(2\.\d+\.))", text)
    cur_ref = "2.1"
    sentences: List[str] = []
    sent_ref: Dict[int, str] = {}

    i = 0
    k = 0
    while i < len(parts):
        chunk = parts[i]
        if re.fullmatch(r"(?:^|\s)(2\.\d+\.)", chunk or ""):
            cur_ref = re.search(r"(2\.\d+)\.", chunk).group(1)
            i += 1
            continue
        chunk = (chunk or "").strip()
        if chunk:
            for s in sentence_split(chunk):
                s2 = s.strip()
                # drop bare headers like "2.1."
                if re.fullmatch(r"2\.\d+\.", s2): 
                    continue
                sentences.append(s2)
                sent_ref[k] = cur_ref
                k += 1
        i += 1

    return sentences, sent_ref

def sanitize_summary_text(text: str) -> str:
    if not text: return ""
    t = re.sub(r"^\s*Specialty\s+Summary\.?\s*", "", text.strip(), flags=re.I)
    return collapse_soft_linebreaks(t)

def parse_doc_block(block: str) -> Optional[AFSCEntry]:
    title = title_from_block(block)
    if title in IGNORED_TITLES:
        return None

    cem, ladders = parse_ladders(block)
    summary = extract_section(block, RE_SUMMARY, [RE_DUTIES, RE_QUALS, RE_SHRED])
    duties  = extract_section(block, RE_DUTIES , [RE_QUALS, RE_SHRED, RE_SUMMARY])
    quals   = extract_section(block, RE_QUALS  , [RE_SHRED, RE_DUTIES, RE_SUMMARY])

    # Heuristic: skip non-AFSC blocks (no ladders and no duties/summary)
    if not ladders and not RE_DUTIES.search(block) and not RE_SUMMARY.search(block):
        return None

    entry = AFSCEntry(
        cem_code=cem,
        title=title,
        summary=sanitize_summary_text(summary),
        duties=duties,
        quals=quals,
        ladders=ladders,
        shred_block=block,
    )
    return entry

def build_doc_id(title: str, ladder_codes: List[str], cem: Optional[str]) -> str:
    base = slugify(title if title and title != "(UNKNOWN TITLE)" else "unknown-title")
    salt = ";".join(ladder_codes[:4]) or (cem or "no-afsc")
    return f"{base}-{sha1(base+'|'+salt)[:10]}"

def parse_experience_hints(quals_text: str) -> List[Tuple[str, Optional[str], str, str]]:
    out = []
    if not quals_text: return out
    ex_start = re.search(r"3\.4\.\s*Experience\.", quals_text, flags=re.I)
    scope = quals_text[ex_start.start():] if ex_start else quals_text
    for m in RE_EXPERIENCE_LINE.finditer(scope):
        code = m.group(1)
        sd = ladder_digit_from_code(code)
        rest = m.group(2).strip()
        verbs = sorted(set(v for v in VERB_MAP if re.search(rf"\b{re.escape(v)}\b", rest.lower())))
        out.append((m.group(0).split(".")[0]+".", sd, ";".join(verbs), rest))
    return out

# -----------------------------
# Main
# -----------------------------

def run(pdf_path: str, out_dir: str, include_experience: bool):
    os.makedirs(out_dir, exist_ok=True)
    pages = extract_pages(pdf_path)
    big = "\n".join(pages)
    blocks = cut_docs(big)

    afsc_docs_rows = []
    summaries_rows = []
    duties_rows = []
    duty_skill_rows = []
    canon_map: Dict[str, Dict[str, str]] = {}
    embeds_rows = []
    shred_rows = []
    exp_rows = []

    for block in blocks:
        entry = parse_doc_block(block)
        if not entry: 
            continue

        codes, digits_sorted, labels_sorted = unify_ladder(entry.ladders)
        doc_id = build_doc_id(entry.title or "(UNKNOWN TITLE)", codes, entry.cem_code)

        afsc_docs_rows.append({
            "doc_id": doc_id,
            "cem_code": entry.cem_code or "",
            "afsc_title": entry.title or "(UNKNOWN TITLE)",
            "afsc_codes": ";".join(codes),
            "afsc_families": ";".join(sorted(set(afsc_family(c) for c in codes))) if codes else "",
            "skill_digits": ";".join(digits_sorted),
            "skill_labels": ";".join(labels_sorted),
        })

        if entry.summary:
            summaries_rows.append({
                "doc_id": doc_id,
                "section": "summary",
                "section_ref": "1",
                "text": entry.summary,
            })

        shreds = parse_shredouts(entry.shred_block)
        for suf, txt in shreds:
            shred_rows.append({"doc_id": doc_id, "suffix": suf, "primary_aircraft": txt})

        if entry.duties:
            sentences, sentref = duties_to_sentences(entry.duties)
            for idx, s in enumerate(sentences):
                can_id = canonical_duty_id(s)
                duties_rows.append({
                    "doc_id": doc_id,
                    "ref": sentref.get(idx, "2.1"),
                    "sent_index": idx,
                    "sentence_text": s,
                    "canonical_duty_id": can_id,
                })
                if can_id not in canon_map:
                    canon_map[can_id] = {
                        "canonical_duty_id": can_id,
                        "sentence_text": s,
                        "sentence_text_norm": default_process(s.lower()),
                    }
                digs, reason, conf = infer_skills_for_sentence(s)
                for d in digs:
                    duty_skill_rows.append({
                        "doc_id": doc_id,
                        "ref": sentref.get(idx, "2.1"),
                        "sent_index": idx,
                        "skill_digit": d,
                        "inference_reason": reason,
                        "confidence": conf,
                    })
                # embeddings row
                title_lbl = entry.title or "(UNKNOWN TITLE)"
                text_emb = f"[{title_lbl} | Duty {sentref.get(idx, '2.1')} | Skills: {','.join(str(x) for x in sorted(set(digs)))}] {s}"
                meta = {
                    "doc_id": doc_id,
                    "afsc_title": title_lbl,
                    "afsc_codes": codes,
                    "ref": sentref.get(idx, "2.1"),
                    "sent_index": idx,
                    "skill_digits": [str(x) for x in sorted(set(digs))]
                }
                embeds_rows.append({
                    "doc_id": doc_id,
                    "ref": sentref.get(idx, "2.1"),
                    "sent_index": idx,
                    "text_for_embedding": text_emb,
                    "metadata_json": json.dumps(meta, ensure_ascii=False),
                })

        if include_experience and entry.quals:
            for section_ref, sd, verbs, raw in parse_experience_hints(entry.quals):
                exp_rows.append({
                    "doc_id": doc_id,
                    "section_ref": section_ref,
                    "target_skill_digit": sd or "",
                    "verbs_extracted": verbs,
                    "raw_text": raw,
                })

    # write
    pd.DataFrame(afsc_docs_rows).to_csv(os.path.join(out_dir, "afsc_docs.csv"), index=False)
    pd.DataFrame(summaries_rows).to_csv(os.path.join(out_dir, "afsc_summaries.csv"), index=False)
    pd.DataFrame(duties_rows).to_csv(os.path.join(out_dir, "duty_sentences.csv"), index=False)
    pd.DataFrame(duty_skill_rows).to_csv(os.path.join(out_dir, "duty_skill_map.csv"), index=False)
    pd.DataFrame(list(canon_map.values())).to_csv(os.path.join(out_dir, "canonical_duties.csv"), index=False)
    pd.DataFrame(embeds_rows).to_csv(os.path.join(out_dir, "embeddings_input.csv"), index=False)
    pd.DataFrame(shred_rows).to_csv(os.path.join(out_dir, "shredouts.csv"), index=False)
    if include_experience:
        pd.DataFrame(exp_rows).to_csv(os.path.join(out_dir, "afsc_experience_hints.csv"), index=False)

    print(f"Done. CSVs written to: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-experience", action="store_true")
    args = ap.parse_args()
    run(args.pdf, args.out, include_experience=args.include_experience)
