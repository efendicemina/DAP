import os
import re
import json
import hashlib
import logging
import threading
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag, urlencode
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from tqdm import tqdm
import trafilatura
from pypdf import PdfReader
from docx import Document
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================
# SETTINGS
# ============================

START_URLS = ["https://www.mpr.gov.ba/bs"]
ALLOWED_DOMAIN = "mpr.gov.ba"

OUTPUT_DIR = Path("mpr_dataset_audit")
FILES_DIR = OUTPUT_DIR / "files"
REPORTS_DIR = OUTPUT_DIR / "reports"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
FILES_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

MAX_PAGES = 1500
MAX_WORKERS = 8
TIMEOUT = 25
CHECKPOINT_EVERY = 50

DOCUMENT_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".txt", ".rtf"
}

KEEP_SECTIONS = [
    "registracije", "registracija", "udruzenja", "udruženja", "fondacije",
    "obrasci", "formulari", "potrebna-dokumentacija", "administrativna-taksa",
    "kontakt", "registar", "pravna-pomoc", "pravna-pomoć",
    "besplatna-pravna-pomoc", "besplatna-pravna-pomoć",
    "zakoni", "propisi", "pravilnici", "podzakonski-akti",
    "medjunarodna-pravna-pomoc", "međunarodna-pravna-pomoć",
    "alimentacije", "otmice-djece", "ispiti", "pravosudni-ispit",
    "strucni-upravni-ispit", "stručni-upravni-ispit",
    "zospi", "pristup-informacijama", "sudski-tumaci", "sudski-tumači",
    "notari"
]

DROP_SECTIONS = [
    "vijesti", "novosti", "saopstenja", "saopštenja", "press",
    "galerija", "foto", "video", "tenderi", "javne-nabavke",
    "javni-oglasi", "konkursi", "projekti", "strategije",
    "eu4justice", "transparentnost", "oglasna-ploca", "oglasna-ploča",
    "izvjestaj-o-radu", "izvještaj-o-radu", "godisnji-izvjestaj",
    "godišnji-izvještaj", "budzet", "budžet", "program-rada"
]

RELEVANT_TERMS = [
    "zahtjev", "obrazac", "registracija", "rješenje", "zakon",
    "pravilnik", "postupak", "nadležnost", "ministarstvo",
    "udruženje", "fondacija", "notar", "sudski tumač",
    "pravna pomoć", "dokumentacija", "rok", "taksa",
    "ispit", "registar", "formular", "podnosi"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Academic NLP Dataset Scraper; +student-project)"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)
thread_local = threading.local()


# ============================
# SESSION
# ============================

def get_session():
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        session.headers.update(HEADERS)

        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=50,
            pool_maxsize=50
        )

        session.mount("https://", adapter)
        session.mount("http://", adapter)
        thread_local.session = session

    return thread_local.session


# ============================
# HELPERS
# ============================

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text_for_match(text):
    text = text.lower()
    text = text.replace("č", "c").replace("ć", "c")
    text = text.replace("š", "s").replace("đ", "dj").replace("ž", "z")
    return text


def normalize_url(url):
    url, _ = urldefrag(url)
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return None

    if not parsed.netloc.endswith(ALLOWED_DOMAIN):
        return None

    path = parsed.path.rstrip("/") or "/"

    query = ""
    if parsed.query:
        params = []

        for q in parsed.query.split("&"):
            if "=" in q:
                k, v = q.split("=", 1)
            else:
                k, v = q, ""

            if not k.startswith("utm_") and k.lower() not in ["ref", "fbclid"]:
                params.append((k, v))

        if params:
            query = urlencode(params)

    normalized = f"{parsed.scheme}://{parsed.netloc}{path}"

    if query:
        normalized += f"?{query}"

    return normalized


def is_document_url(url):
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in DOCUMENT_EXTENSIONS)


def get_extension(url):
    return Path(urlparse(url).path).suffix.lower()


def safe_filename(url):
    parsed = urlparse(url)
    name = os.path.basename(parsed.path)

    if not name:
        name = hashlib.md5(url.encode()).hexdigest() + ".html"

    qhash = hashlib.md5((parsed.query or "").encode()).hexdigest()[:8]
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)

    if "." not in name:
        name += ".html"

    return f"{qhash}_{name}"


def extract_title(soup, fallback=""):
    h1 = soup.find("h1")
    if h1:
        return clean_text(h1.get_text(" "))

    title = soup.find("title")
    if title:
        return clean_text(title.get_text(" "))

    return fallback


def get_category_from_url(url):
    path = urlparse(url).path
    parts = [p for p in path.split("/") if p]

    if len(parts) >= 2:
        return " / ".join(parts[:4])

    return ""


def infer_category(url, title="", parent_title=""):
    combined = normalize_text_for_match(" ".join([url, title, parent_title]))

    category_rules = {
        "registracija": ["registracija", "registracije"],
        "obrasci": ["obrasci", "obrazac", "formulari", "formular"],
        "zakoni_i_propisi": ["zakoni", "zakon", "propisi", "pravilnici", "podzakonski"],
        "pravna_pomoc": ["pravna-pomoc", "pravna pomoc", "besplatna-pravna"],
        "notari": ["notari", "notar"],
        "sudski_tumaci": ["sudski-tumaci", "sudski tumaci", "tumaci"],
        "ispiti": ["ispiti", "pravosudni", "upravni-ispit"],
        "registri": ["registar", "registri"],
        "kontakt_i_nadleznosti": ["kontakt", "nadleznost", "nadleznosti"]
    }

    for category, terms in category_rules.items():
        if any(term in combined for term in terms):
            return category

    return "ostalo"


def classify_url(url):
    normalized = normalize_text_for_match(url)

    if any(term in normalized for term in map(normalize_text_for_match, DROP_SECTIONS)):
        return "drop_by_url"

    if any(term in normalized for term in map(normalize_text_for_match, KEEP_SECTIONS)):
        return "keep_by_url"

    return "unknown_by_url"


def classify_content(url, title, text, document_type, parent_title=""):
    normalized_all = normalize_text_for_match(" ".join([url, title, parent_title, text[:2000]]))

    url_status = classify_url(url)

    if url_status == "drop_by_url":
        return False, "news_or_temporary_content", "irrelevant"

    if document_type in ["pdf", "docx", "doc", "xlsx", "xls"]:
        if url_status == "keep_by_url":
            return True, "relevant_document_url", "ok"

        if any(normalize_text_for_match(t) in normalized_all for t in RELEVANT_TERMS):
            return True, "relevant_document_content", "ok"

        return False, "document_not_relevant_enough", "unknown_review_needed"

    if len(text) < 250:
        return False, "too_short", "too_short"

    if url_status == "keep_by_url":
        return True, "relevant_url_section", "ok"

    if any(normalize_text_for_match(t) in normalized_all for t in RELEVANT_TERMS):
        return True, "relevant_content_terms", "ok"

    return False, "not_relevant_enough", "unknown_review_needed"


# ============================
# EXTRACTION
# ============================

def extract_html_text(html):
    try:
        extracted = trafilatura.extract(html)
        if extracted:
            return clean_text(extracted)
    except Exception:
        pass

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    return clean_text(soup.get_text(" "))


def extract_pdf_text(path, max_pages=30):
    try:
        reader = PdfReader(path)
        pages = []

        for i, page in enumerate(reader.pages[:max_pages]):
            txt = page.extract_text() or ""

            if txt.strip():
                pages.append(f"[PAGE {i + 1}] {txt}")

        return clean_text("\n".join(pages))
    except Exception:
        return ""


def extract_docx_text(path):
    try:
        doc = Document(path)
        return clean_text("\n".join([p.text for p in doc.paragraphs]))
    except Exception:
        return ""


def extract_excel_text(path, max_rows=100):
    try:
        excel = pd.ExcelFile(path)
        parts = []

        for sheet in excel.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet, nrows=max_rows)
            df = df.dropna(how="all")

            if df.empty:
                continue

            parts.append(f"Sheet: {sheet}")
            parts.append(df.astype(str).to_csv(index=False))

        return clean_text("\n".join(parts))
    except Exception:
        return ""


def extract_file_text(path):
    ext = path.suffix.lower()

    if ext == ".pdf":
        return extract_pdf_text(path)

    if ext == ".docx":
        return extract_docx_text(path)

    if ext in [".xls", ".xlsx"]:
        return extract_excel_text(path)

    if ext in [".txt", ".rtf"]:
        try:
            return clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return ""

    return ""


def extract_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    doc_links = []

    for a in soup.find_all("a", href=True):
        anchor_text = clean_text(a.get_text(" "))
        next_url = urljoin(base_url, a["href"])
        next_url = normalize_url(next_url)

        if not next_url:
            continue

        item = {
            "url": next_url,
            "anchor_text": anchor_text
        }

        if is_document_url(next_url):
            doc_links.append(item)
        else:
            links.append(item)

    return links, doc_links


def content_hash(text):
    return hashlib.md5(clean_text(text).encode("utf-8")).hexdigest()


# ============================
# SCRAPER
# ============================

def scrape_task(task):
    url = task["url"]
    parent_url = task.get("parent_url", "")
    parent_title = task.get("parent_title", "")
    anchor_text = task.get("anchor_text", "")

    try:
        session = get_session()
        response = session.get(url, timeout=TIMEOUT)

        if response.status_code != 200:
            return None, [], {
                "url": url,
                "status": "failed",
                "http_status": response.status_code
            }

        content_type = response.headers.get("Content-Type", "").lower()
        ext = get_extension(url)

        is_doc = (
            is_document_url(url)
            or "application/pdf" in content_type
            or "officedocument" in content_type
            or "msword" in content_type
            or "spreadsheet" in content_type
        )

        if is_doc:
            filename = safe_filename(url)
            file_path = FILES_DIR / filename
            file_path.write_bytes(response.content)

            document_type = ext.replace(".", "") if ext else "document"
            text = extract_file_text(file_path)
            title = anchor_text or filename

            ocr_required = document_type == "pdf" and len(text) < 100
            processing_status = "needs_ocr" if ocr_required else "text_extracted"

            category = infer_category(url, title, parent_title)
            keep, reason, quality_flag = classify_content(
                url=url,
                title=title,
                text=text,
                document_type=document_type,
                parent_title=parent_title
            )

            if ocr_required and classify_url(url) == "keep_by_url":
                keep = True
                reason = "relevant_pdf_but_ocr_required"
                quality_flag = "needs_ocr"

            record = {
                "source_url": url,
                "title": title,
                "category": category,
                "document_type": document_type,
                "content_type": content_type,
                "file_path": str(file_path),
                "parent_url": parent_url,
                "parent_title": parent_title,
                "anchor_text": anchor_text,
                "text": text,
                "text_length": len(text),
                "ocr_required": ocr_required,
                "processing_status": processing_status,
                "keep": keep,
                "reason": reason,
                "quality_flag": quality_flag,
                "hash": content_hash(text) if text else ""
            }

            return record, [], {
                "url": url,
                "status": "ok",
                "http_status": 200
            }

        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        title = extract_title(soup, fallback=url)
        text = extract_html_text(html)

        links, doc_links = extract_links(html, url)

        document_type = "html"
        category = infer_category(url, title, parent_title)

        keep, reason, quality_flag = classify_content(
            url=url,
            title=title,
            text=text,
            document_type=document_type,
            parent_title=parent_title
        )

        if len(text) < 250 and len(doc_links) > 0:
            keep = True
            reason = "index_page_with_documents"
            quality_flag = "index_page_only"

        record = {
            "source_url": url,
            "title": title,
            "category": category,
            "document_type": "html",
            "content_type": content_type,
            "file_path": "",
            "parent_url": parent_url,
            "parent_title": parent_title,
            "anchor_text": anchor_text,
            "text": text,
            "text_length": len(text),
            "linked_documents_count": len(doc_links),
            "ocr_required": False,
            "processing_status": "text_extracted",
            "keep": keep,
            "reason": reason,
            "quality_flag": quality_flag,
            "hash": content_hash(text) if text else ""
        }

        next_tasks = []

        for link in links:
            next_tasks.append({
                "url": link["url"],
                "parent_url": url,
                "parent_title": title,
                "anchor_text": link["anchor_text"]
            })

        for link in doc_links:
            next_tasks.append({
                "url": link["url"],
                "parent_url": url,
                "parent_title": title,
                "anchor_text": link["anchor_text"]
            })

        return record, next_tasks, {
            "url": url,
            "status": "ok",
            "http_status": 200
        }

    except Exception as e:
        return None, [], {
            "url": url,
            "status": "error",
            "error": str(e)
        }


# ============================
# SAVE + REPORTING
# ============================

def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_chunks(records, max_chars=1400, overlap=200):
    chunks = []
    seen = set()

    for r in records:
        if not r.get("keep"):
            continue

        if r.get("ocr_required"):
            continue

        text = clean_text(r.get("text", ""))

        if len(text) < 250:
            continue

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + max_chars
            chunk = text[start:end].strip()

            if len(chunk) > 150:
                h = content_hash(chunk)

                if h not in seen:
                    seen.add(h)

                    chunks.append({
                        "chunk_id": f"{hashlib.md5(r['source_url'].encode()).hexdigest()}_{chunk_index}",
                        "source_url": r["source_url"],
                        "title": r["title"],
                        "category": r["category"],
                        "document_type": r["document_type"],
                        "parent_url": r.get("parent_url", ""),
                        "parent_title": r.get("parent_title", ""),
                        "chunk_index": chunk_index,
                        "text": chunk
                    })

                    chunk_index += 1

            start = end - overlap

            if start < 0:
                start = end

    return chunks


def checkpoint(records, events):
    save_jsonl(OUTPUT_DIR / "all_records_checkpoint.jsonl", records)
    pd.DataFrame(records).to_csv(OUTPUT_DIR / "all_records_checkpoint.csv", index=False)
    pd.DataFrame(events).to_csv(OUTPUT_DIR / "crawl_events_checkpoint.csv", index=False)


def generate_reports(records, events, chunks):
    df = pd.DataFrame(records)

    if df.empty:
        print("No records to report.")
        return

    all_csv = OUTPUT_DIR / "all_records.csv"
    kept_csv = OUTPUT_DIR / "kept_records.csv"
    dropped_csv = OUTPUT_DIR / "dropped_records.csv"
    needs_review_csv = OUTPUT_DIR / "needs_review.csv"
    needs_ocr_csv = OUTPUT_DIR / "needs_ocr.csv"
    chunks_jsonl = OUTPUT_DIR / "chunks.jsonl"

    df.to_csv(all_csv, index=False)
    df[df["keep"] == True].to_csv(kept_csv, index=False)
    df[df["keep"] == False].to_csv(dropped_csv, index=False)
    df[df["quality_flag"].isin(["unknown_review_needed", "index_page_only", "too_short"])].to_csv(needs_review_csv, index=False)
    df[df["ocr_required"] == True].to_csv(needs_ocr_csv, index=False)

    save_jsonl(chunks_jsonl, chunks)
    pd.DataFrame(events).to_csv(OUTPUT_DIR / "crawl_events.csv", index=False)

    # Stats
    stats = {
        "total_records": len(df),
        "kept_records": int((df["keep"] == True).sum()),
        "dropped_records": int((df["keep"] == False).sum()),
        "ocr_required": int((df["ocr_required"] == True).sum()),
        "chunks_created": len(chunks),
        "avg_text_length": float(df["text_length"].mean()),
        "total_text_chars": int(df["text_length"].sum())
    }

    with open(OUTPUT_DIR / "summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Charts
    chart_paths = []

    def save_bar(series, title, filename):
        if series.empty:
            return

        plt.figure(figsize=(10, 5))
        series.plot(kind="bar")
        plt.title(title)
        plt.ylabel("Broj")
        plt.tight_layout()
        path = REPORTS_DIR / filename
        plt.savefig(path, dpi=150)
        plt.close()
        chart_paths.append(path)

    save_bar(df["keep"].value_counts(), "Keep vs Drop", "keep_vs_drop.png")
    save_bar(df["document_type"].value_counts().head(15), "Tipovi dokumenata", "document_types.png")
    save_bar(df["category"].value_counts().head(15), "Kategorije", "categories.png")
    save_bar(df["quality_flag"].value_counts().head(15), "Quality flags", "quality_flags.png")
    save_bar(df["reason"].value_counts().head(15), "Razlozi odluke", "reasons.png")

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>MPR Dataset Audit Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #fafafa; }}
            .card {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
            h1, h2 {{ color: #222; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 14px; }}
            th {{ background: #f0f0f0; }}
            img {{ max-width: 900px; width: 100%; border-radius: 8px; }}
            code {{ background: #eee; padding: 3px 6px; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <h1>MPR Dataset Audit Report</h1>

        <div class="card">
            <h2>Summary</h2>
            <table>
                {''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k, v in stats.items())}
            </table>
        </div>

        <div class="card">
            <h2>Generated files</h2>
            <ul>
                <li><code>all_records.csv</code></li>
                <li><code>kept_records.csv</code></li>
                <li><code>dropped_records.csv</code></li>
                <li><code>needs_review.csv</code></li>
                <li><code>needs_ocr.csv</code></li>
                <li><code>chunks.jsonl</code></li>
                <li><code>summary_stats.json</code></li>
            </ul>
        </div>

        {''.join(f'<div class="card"><img src="reports/{p.name}"></div>' for p in chart_paths)}
    </body>
    </html>
    """

    with open(OUTPUT_DIR / "audit_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("\nDONE")
    print(f"Total records: {stats['total_records']}")
    print(f"Kept: {stats['kept_records']}")
    print(f"Dropped: {stats['dropped_records']}")
    print(f"OCR required: {stats['ocr_required']}")
    print(f"Chunks created: {stats['chunks_created']}")
    print(f"Report: {OUTPUT_DIR / 'audit_report.html'}")


# ============================
# MAIN CRAWLER
# ============================

def crawl():
    visited = set()
    queued = set()
    records = []
    events = []

    queue = deque()

    for url in START_URLS:
        normalized = normalize_url(url)
        queue.append({
            "url": normalized,
            "parent_url": "",
            "parent_title": "",
            "anchor_text": ""
        })
        queued.add(normalized)

    pbar = tqdm(total=MAX_PAGES, desc="Scraping", unit="url")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while queue and len(visited) < MAX_PAGES:
            batch = []

            while queue and len(batch) < MAX_WORKERS * 3 and len(visited) + len(batch) < MAX_PAGES:
                task = queue.popleft()
                url = task["url"]

                if url in visited:
                    continue

                visited.add(url)
                batch.append(task)

            futures = {executor.submit(scrape_task, task): task for task in batch}

            for future in as_completed(futures):
                record, next_tasks, event = future.result()
                events.append(event)

                if record:
                    records.append(record)

                for task in next_tasks:
                    url = task["url"]

                    if url not in visited and url not in queued:
                        queued.add(url)
                        queue.append(task)

                pbar.update(1)

                kept = sum(1 for r in records if r.get("keep") is True)
                ocr = sum(1 for r in records if r.get("ocr_required") is True)

                pbar.set_postfix({
                    "records": len(records),
                    "kept": kept,
                    "ocr": ocr,
                    "queue": len(queue)
                })

                if len(records) > 0 and len(records) % CHECKPOINT_EVERY == 0:
                    checkpoint(records, events)

    pbar.close()

    # remove duplicates by hash + url fallback
    deduped = []
    seen_hashes = set()
    seen_urls = set()

    for r in records:
        h = r.get("hash", "")
        url = r.get("source_url", "")

        if h and h in seen_hashes:
            r["keep"] = False
            r["reason"] = "duplicate_content"
            r["quality_flag"] = "duplicate"
            deduped.append(r)
            continue

        if url in seen_urls:
            r["keep"] = False
            r["reason"] = "duplicate_url"
            r["quality_flag"] = "duplicate"
            deduped.append(r)
            continue

        if h:
            seen_hashes.add(h)

        seen_urls.add(url)
        deduped.append(r)

    chunks = make_chunks(deduped)
    generate_reports(deduped, events, chunks)


if __name__ == "__main__":
    crawl()