import re
import json
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUTPUT_DIR = Path("mpr_dataset_v4")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

SITEMAP_CSV = Path("mpr_dataset_v3_fixed/selected_sitemap_links.csv")

TIMEOUT = 25
CHUNK_SIZE = 1300
CHUNK_OVERLAP = 180
MIN_TEXT_LENGTH = 120

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MPR Dataset v4; student-project)"
}

MANUAL_URLS = [
    # Udruženja
    ("https://mpr.gov.ba/bs/kako-osnovati-udruzenje", "registracija", "udruzenja", "high"),
    ("https://mpr.gov.ba/bs/potrebna-dokumentacija", "registracija", "udruzenja", "high"),
    ("https://mpr.gov.ba/bs/obrasci", "obrasci", "udruzenja", "high"),
    ("https://mpr.gov.ba/bs/administrativna-taksa55", "registracija", "udruzenja", "high"),
    ("https://mpr.gov.ba/bs/upis-promjena-u-registru-udruzenja-ili-fondacija", "registracija", "udruzenja", "high"),
    ("https://mpr.gov.ba/bs/brisanja-iz-registra-po-zahtjevu-udruzenja-i-fondacije", "registracija", "udruzenja", "high"),
    ("https://mpr.gov.ba/bs/formulari97", "obrasci", "udruzenja", "high"),
    ("https://mpr.gov.ba/bs/administrativna-taksa84", "registracija", "udruzenja", "high"),
    ("https://mpr.gov.ba/bs/kontakt63", "kontakt_i_nadleznosti", "udruzenja", "medium"),

    # Fondacije
    ("https://mpr.gov.ba/bs/kako-osnovati-fondaciju", "registracija", "fondacije", "high"),
    ("https://mpr.gov.ba/bs/registracija-fondacije", "registracija", "fondacije", "high"),
    ("https://mpr.gov.ba/bs/administrativna-taksa9", "registracija", "fondacije", "high"),

    # Strane NVO / pravna lica / crkve
    ("https://mpr.gov.ba/bs/administrativna-taksa21", "registracija", "strane_nvo", "medium"),
    ("https://mpr.gov.ba/bs/administrativne-takse51", "registracija", "pravna_lica", "medium"),
    ("https://mpr.gov.ba/bs/administrativne-takse87", "registracija", "crkve_vjerske_zajednice", "medium"),

    # Pravosudni ispit
    ("https://mpr.gov.ba/bs/pravosudni-ispit", "ispiti", "pravosudni_ispit", "high"),
    ("https://mpr.gov.ba/bs/uslovi-za-polaganje-ispita-", "ispiti", "pravosudni_ispit", "high"),
    ("https://mpr.gov.ba/bs/prijava-polaganja-pravosudnog-ispita", "ispiti", "pravosudni_ispit", "high"),
    ("https://mpr.gov.ba/bs/nacin-polaganja-pravosudnog-ispita", "ispiti", "pravosudni_ispit", "high"),
    ("https://mpr.gov.ba/bs/rjesenje-o-imenovanju-povjerenstva-za-polaganje-pravosudnog-ispita-na-nivou-bosne-i-hercegovine", "ispiti", "pravosudni_ispit", "medium"),
    ("https://mpr.gov.ba/bs/troskovi-polaganja-ispita", "ispiti", "pravosudni_ispit", "high"),
    ("https://mpr.gov.ba/bs/kontakt", "kontakt_i_nadleznosti", "pravosudni_ispit", "medium"),
    ("https://mpr.gov.ba/bs/novi-ispitni-termini", "ispiti", "pravosudni_ispit", "medium"),
    ("https://mpr.gov.ba/bs/literatura", "ispiti", "pravosudni_ispit", "medium"),

    # Pravna pomoć
    ("https://mpr.gov.ba/bs/besplatna-pravna-pomoc", "pravna_pomoc", "besplatna_pravna_pomoc", "high"),
    ("https://mpr.gov.ba/bs/ured-za-pruzanje-besplatne-pravne-pomoci", "pravna_pomoc", "besplatna_pravna_pomoc", "high"),

    # Zakoni
    ("https://mpr.gov.ba/bs/zakoni-i-drugi-propisi", "zakoni_i_propisi", "zakoni", "high"),
    ("https://mpr.gov.ba/bs/zakoni-ministarstva-i-drugi-vazniji-zakoni", "zakoni_i_propisi", "zakoni", "high"),
    ("https://mpr.gov.ba/bs/podzakonski-akti", "zakoni_i_propisi", "podzakonski_akti", "high"),
]

def make_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

session = make_session()

def clean_text(text):
    if not text:
        return ""
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def norm(text):
    text = clean_text(text).lower()
    return (
        text.replace("č", "c")
        .replace("ć", "c")
        .replace("š", "s")
        .replace("ž", "z")
        .replace("đ", "dj")
    )

def hash_text(text):
    return hashlib.md5(clean_text(text).encode("utf-8")).hexdigest()

def extract_title(html, fallback):
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    if h1:
        return clean_text(h1.get_text(" "))
    title = soup.find("title")
    if title:
        return clean_text(title.get_text(" "))
    return fallback

def extract_html_text(html):
    try:
        extracted = trafilatura.extract(html)
        if extracted and len(extracted) > 80:
            return clean_text(extracted)
    except Exception:
        pass

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    return clean_text(soup.get_text(" "))

def infer_category(url, title, text):
    u = norm(url)
    t = norm(title)
    combined = norm(url + " " + title + " " + text[:800])

    # najprije specifične oblasti
    if any(x in u for x in ["pravosudni-ispit", "strucni-upravni-ispit", "polaganja-ispita", "ispitni-termini"]) or "ispit" in t:
        return "ispiti"

    if any(x in u for x in ["udruzenj", "fondacij", "nvo", "registar", "administrativna-taksa", "potrebna-dokumentacija", "upis-promjena", "brisanja-iz-registra"]):
        return "registracija"

    if any(x in u for x in ["besplatna-pravna-pomoc", "pravna-pomoc", "alimentacije", "otmice-djece"]):
        return "pravna_pomoc"

    if any(x in u for x in ["zakoni", "propis", "podzakonski", "ustav", "ugovori", "konvencije"]):
        return "zakoni_i_propisi"

    if any(x in u for x in ["kontakt", "nadleznosti", "organizacija", "sektor", "ured"]):
        return "kontakt_i_nadleznosti"

    # obrasci tek nakon ovoga
    if any(x in u for x in ["obrasci", "formulari", "formular", "obrazac"]):
        return "obrasci"

    if "registar" in combined:
        return "registri"

    return "ostalo"

def infer_subsection(url, title):
    u = norm(url + " " + title)

    if "pravosudni" in u:
        return "pravosudni_ispit"
    if "strucni-upravni" in u or "strucni upravni" in u:
        return "strucni_upravni_ispit"
    if "udruzenj" in u:
        return "udruzenja"
    if "fondacij" in u:
        return "fondacije"
    if "besplatna-pravna-pomoc" in u:
        return "besplatna_pravna_pomoc"
    if "alimentacij" in u:
        return "alimentacije"
    if "otmica" in u:
        return "otmice_djece"
    if "zakon" in u or "propis" in u:
        return "zakoni"
    return ""

def load_urls():
    items = []

    if SITEMAP_CSV.exists():
        df = pd.read_csv(SITEMAP_CSV)
        for _, row in df.iterrows():
            url = str(row.get("url", "")).strip()
            title = str(row.get("title", "")).strip()
            if url:
                items.append({
                    "source_url": url,
                    "seed_title": title,
                    "manual_category": "",
                    "manual_subsection": "",
                    "priority": "normal",
                    "source": "sitemap"
                })

    for url, category, subsection, priority in MANUAL_URLS:
        items.append({
            "source_url": url,
            "seed_title": "",
            "manual_category": category,
            "manual_subsection": subsection,
            "priority": priority,
            "source": "manual"
        })

    # dedupe, manual wins
    by_url = {}
    for item in items:
        url = item["source_url"]
        if url not in by_url or item["source"] == "manual":
            by_url[url] = item

    return list(by_url.values())

def scrape(item):
    url = item["source_url"]
    try:
        r = session.get(url, timeout=25)
        if r.status_code != 200:
            return None

        html = r.text
        title = extract_title(html, item.get("seed_title", "") or url)
        text = extract_html_text(html)

        category = item.get("manual_category") or infer_category(url, title, text)
        subsection = item.get("manual_subsection") or infer_subsection(url, title)

        return {
            "source_url": url,
            "title": title,
            "category": category,
            "subsection": subsection,
            "priority": item.get("priority", "normal"),
            "source": item.get("source", "sitemap"),
            "document_type": "html",
            "text": text,
            "text_length": len(text),
            "keep": len(text) >= MIN_TEXT_LENGTH,
            "hash": hash_text(text) if text else ""
        }
    except Exception as e:
        return {
            "source_url": url,
            "title": item.get("seed_title", ""),
            "category": item.get("manual_category", "ostalo"),
            "subsection": item.get("manual_subsection", ""),
            "priority": item.get("priority", "normal"),
            "source": item.get("source", "sitemap"),
            "document_type": "html",
            "text": "",
            "text_length": 0,
            "keep": False,
            "error": str(e),
            "hash": ""
        }

def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", clean_text(text))

def make_chunks(records):
    chunks = []
    seen = set()

    for r in records:
        if not r.get("keep"):
            continue

        text = clean_text(r["text"])
        if len(text) < MIN_TEXT_LENGTH:
            continue

        sentences = split_sentences(text)
        current = ""
        idx = 0

        for sent in sentences:
            if len(current) + len(sent) <= CHUNK_SIZE:
                current += " " + sent
            else:
                chunk = clean_text(current)
                normalized = norm(chunk)

                if len(chunk) >= MIN_TEXT_LENGTH and len(chunk.split()) >= 30 and normalized not in seen:
                    seen.add(normalized)
                    chunks.append({
                        "chunk_id": f"{hash_text(r['source_url'])}_{idx}",
                        "source_url": r["source_url"],
                        "title": r["title"],
                        "category": r["category"],
                        "subsection": r["subsection"],
                        "priority": r["priority"],
                        "document_type": r["document_type"],
                        "chunk_index": idx,
                        "text": chunk
                    })
                    idx += 1

                current = chunk[-CHUNK_OVERLAP:] + " " + sent

        chunk = clean_text(current)
        normalized = norm(chunk)

        if len(chunk) >= MIN_TEXT_LENGTH and len(chunk.split()) >= 30 and normalized not in seen:
            seen.add(normalized)
            chunks.append({
                "chunk_id": f"{hash_text(r['source_url'])}_{idx}",
                "source_url": r["source_url"],
                "title": r["title"],
                "category": r["category"],
                "subsection": r["subsection"],
                "priority": r["priority"],
                "document_type": r["document_type"],
                "chunk_index": idx,
                "text": chunk
            })

    return chunks

def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    items = load_urls()
    records = []

    for item in tqdm(items, desc="Scraping v4"):
        rec = scrape(item)
        if rec:
            records.append(rec)

    # dedupe exact same URL
    seen_urls = set()
    unique_records = []
    for r in records:
        if r["source_url"] in seen_urls:
            continue
        seen_urls.add(r["source_url"])
        unique_records.append(r)

    chunks = make_chunks(unique_records)

    records_df = pd.DataFrame(unique_records)
    chunks_df = pd.DataFrame(chunks)

    records_df.to_csv(OUTPUT_DIR / "records_v4.csv", index=False)
    chunks_df.to_csv(OUTPUT_DIR / "chunks_v4.csv", index=False)
    save_jsonl(OUTPUT_DIR / "chunks_v4.jsonl", chunks)

    coverage = {
        "total_records": len(records_df),
        "kept_records": int(records_df["keep"].sum()),
        "total_chunks": len(chunks_df),
        "records_by_category": records_df[records_df["keep"] == True]["category"].value_counts().to_dict(),
        "chunks_by_category": chunks_df["category"].value_counts().to_dict() if not chunks_df.empty else {},
        "chunks_by_subsection": chunks_df["subsection"].value_counts().to_dict() if not chunks_df.empty else {},
        "priority_chunks": chunks_df["priority"].value_counts().to_dict() if not chunks_df.empty else {}
    }

    with open(OUTPUT_DIR / "coverage_v4.json", "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)

    print("DONE")
    print(json.dumps(coverage, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()