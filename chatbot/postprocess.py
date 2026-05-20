import re
import json
import hashlib
from pathlib import Path

import pandas as pd

INPUT_DIR = Path("mpr_dataset_audit")
OUTPUT_DIR = Path("mpr_dataset_final")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

ALL_RECORDS = INPUT_DIR / "all_records.csv"

MIN_CHUNK_CHARS = 350
MAX_CHUNK_CHARS = 900
OVERLAP = 120

DROP_URL_PATTERNS = [
    r"/hr($|/)",
    r"/en($|/)",
    r"/sr($|/)",
    r"/portal/objava/",
    r"/portal\?categorySlug=vijesti",
    r"tenderijavni-oglasi",
    r"javne-nabavke",
    r"tenderi",
    r"konkursi",
    r"galerija",
    r"projekti",
    r"strategije",
    r"eu4justice",
]

DROP_EXACT_URLS = {
    "https://www.mpr.gov.ba/bs",
    "https://mpr.gov.ba/bs",
    "https://www.mpr.gov.ba/bs/",
    "https://mpr.gov.ba/bs/",
}

GOOD_CATEGORIES = {
    "registracija",
    "obrasci",
    "zakoni_i_propisi",
    "pravna_pomoc",
    "notari",
    "sudski_tumaci",
    "ispiti",
    "registri",
    "kontakt_i_nadleznosti",
}

GOOD_TERMS = [
    "zahtjev", "obrazac", "registracija", "preregistracija",
    "udruženje", "udruzenje", "fondacija", "zakon", "pravilnik",
    "propis", "pravna pomoć", "pravna pomoc", "notar", "sudski tumač",
    "sudski tumac", "pravosudni ispit", "stručni upravni ispit",
    "strucni upravni ispit", "registar", "taksa", "dokumentacija",
    "nadležnost", "nadleznost", "rok", "rješenje", "rjesenje",
]

BAD_TEXT_STARTS = [
    "vijesti vijesti",
    "news news",
    "hrvatski bosanski српски english",
]

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_bad_url(url):
    url = str(url)

    if url in DROP_EXACT_URLS:
        return True

    for pattern in DROP_URL_PATTERNS:
        if re.search(pattern, url, flags=re.IGNORECASE):
            return True

    return False

def has_good_terms(text):
    lowered = text.lower()
    return any(term in lowered for term in GOOD_TERMS)

def is_bad_text(text):
    lowered = text.lower().strip()

    if len(lowered) < 250:
        return True

    for start in BAD_TEXT_STARTS:
        if lowered.startswith(start):
            return True

    # ako se riječ "vijesti" ponavlja previše, vjerovatno je news listing
    if lowered.count("vijesti") >= 3 and lowered.count("obrazac") == 0 and lowered.count("zakon") == 0:
        return True

    return False

def decide_final_keep(row):
    url = str(row.get("source_url", ""))
    text = clean_text(row.get("text", ""))
    category = str(row.get("category", ""))
    quality_flag = str(row.get("quality_flag", ""))

    if is_bad_url(url):
        return False, "dropped_by_strict_url_filter"

    if quality_flag in ["duplicate", "irrelevant", "too_short"]:
        return False, f"dropped_quality_{quality_flag}"

    if is_bad_text(text):
        return False, "dropped_bad_or_short_text"

    if category in GOOD_CATEGORIES:
        return True, "kept_good_category"

    if has_good_terms(text):
        return True, "kept_good_terms"

    return False, "dropped_not_useful_for_chatbot"

def chunk_text(text):
    text = clean_text(text)
    chunks = []
    start = 0

    while start < len(text):
        end = start + MAX_CHUNK_CHARS
        chunk = text[start:end].strip()

        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)

        start = end - OVERLAP

        if start <= 0:
            start = end

    return chunks

def hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def make_question_templates(title, category):
    title = title if title and title != "nan" else "ovaj dokument"

    return [
        f"Šta korisnik treba znati o temi: {title}?",
        f"Objasni jednostavno sadržaj dokumenta '{title}'.",
        f"Koje su najvažnije informacije iz oblasti {category}?",
        f"Koji su ključni uslovi, dokumenti ili koraci navedeni u tekstu '{title}'?",
        f"Na osnovu dostupnog teksta, kako pomoći korisniku koji pita za '{title}'?"
    ]

def main():
    df = pd.read_csv(ALL_RECORDS)

    decisions = df.apply(decide_final_keep, axis=1, result_type="expand")
    df["final_keep"] = decisions[0]
    df["final_reason"] = decisions[1]

    final_df = df[df["final_keep"] == True].copy()
    dropped_df = df[df["final_keep"] == False].copy()

    final_df.to_csv(OUTPUT_DIR / "final_kept_records.csv", index=False)
    dropped_df.to_csv(OUTPUT_DIR / "final_dropped_records.csv", index=False)

    chunks = []
    seen_chunks = set()

    for _, row in final_df.iterrows():
        text = clean_text(row.get("text", ""))
        source_url = str(row.get("source_url", ""))
        title = str(row.get("title", ""))
        category = str(row.get("category", ""))
        document_type = str(row.get("document_type", ""))

        for i, chunk in enumerate(chunk_text(text)):
            h = hash_text(chunk)

            if h in seen_chunks:
                continue

            seen_chunks.add(h)

            chunks.append({
                "chunk_id": f"{hash_text(source_url)}_{i}",
                "source_url": source_url,
                "title": title,
                "category": category,
                "document_type": document_type,
                "chunk_index": i,
                "text": chunk
            })

    with open(OUTPUT_DIR / "final_chunks.jsonl", "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    qa_rows = []

    for ch in chunks:
        questions = make_question_templates(ch["title"], ch["category"])

        for q in questions:
            qa_rows.append({
                "instruction": q,
                "input": "",
                "output": ch["text"] + f"\n\nIzvor: {ch['source_url']}",
                "source_url": ch["source_url"],
                "title": ch["title"],
                "category": ch["category"]
            })

    with open(OUTPUT_DIR / "qa_dataset_5000_style.jsonl", "w", encoding="utf-8") as f:
        for row in qa_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    chat_rows = []

    for row in qa_rows:
        chat_rows.append({
            "messages": [
                {
                    "role": "system",
                    "content": "Ti si chatbot Ministarstva pravde BiH. Odgovaraš jasno, jednostavno i isključivo na osnovu dostupnog izvora. Ako informacija nije dovoljna, reci korisniku da provjeri zvanični izvor."
                },
                {
                    "role": "user",
                    "content": row["instruction"]
                },
                {
                    "role": "assistant",
                    "content": row["output"]
                }
            ]
        })

    with open(OUTPUT_DIR / "chat_finetune_dataset.jsonl", "w", encoding="utf-8") as f:
        for row in chat_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "original_records": len(df),
        "final_kept_records": len(final_df),
        "final_dropped_records": len(dropped_df),
        "final_chunks": len(chunks),
        "qa_rows": len(qa_rows),
        "chat_rows": len(chat_rows)
    }

    with open(OUTPUT_DIR / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("DONE")
    print(summary)

if __name__ == "__main__":
    main()