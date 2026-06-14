import re
import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path("mpr_dataset_v5")
DATA_DIR.mkdir(exist_ok=True)

chunks = pd.read_csv("mpr_dataset_v4/chunks_v4.csv")
records = pd.read_csv("mpr_dataset_v4/records_v4.csv")


def norm(x):
    x = str(x).lower()
    x = (
        x.replace("č", "c")
        .replace("ć", "c")
        .replace("š", "s")
        .replace("ž", "z")
        .replace("đ", "dj")
    )
    return x

def detect_page_type(title, url, text):
    s = norm(title + " " + url + " " + text[:800])

    title_n = norm(title)
    url_n = norm(url)

    # najprije URL/title specifične stvari
    if "kontakt" in title_n or "kontakt" in url_n:
        return "kontakt"

    if any(x in title_n or x in url_n for x in ["obrazac", "obrasci", "formular", "formulari"]):
        return "obrazac"

    if any(x in title_n or x in url_n for x in ["administrativna-taksa", "administrativne-takse", "taksa", "takse"]):
        return "taksa"

    if any(x in title_n or x in url_n for x in ["troskovi", "troškovi"]):
        return "troskovi"

    if any(x in title_n or x in url_n for x in ["uslovi", "uvjeti"]):
        return "uslovi"

    if any(x in title_n or x in url_n for x in ["prijava", "podnosenje-zahtjeva", "podnošenje-zahtjeva"]):
        return "prijava"

    if any(x in title_n or x in url_n for x in ["termini", "ispitni-termini"]):
        return "termini"

    if "literatura" in title_n or "literatura" in url_n:
        return "literatura"

    if "program" in title_n or "program" in url_n:
        return "program"

    if "prirucnik" in title_n or "priručnik" in title_n or "prirucnik" in url_n:
        return "prirucnik"

    if any(x in title_n or x in url_n for x in ["promjena", "promjene", "izmjena", "dopuna", "upis-promjena"]):
        return "promjena"

    if any(x in title_n or x in url_n for x in ["brisanje", "brisanja", "prestanak-rada"]):
        return "brisanje"

    if any(x in title_n or x in url_n for x in ["registar", "registri", "izvod-iz-registra", "javnost-registra"]):
        return "registar"

    if any(x in title_n or x in url_n for x in ["potrebna-dokumentacija", "dokumentacija"]):
        return "dokumentacija"

    if "osnovne-informacije" in url_n or "osnovne informacije" in title_n:
        return "osnovne_info"
    
    if any(x in title_n or x in url_n for x in [
        "osnovati-udruzenje",
        "osnovati udruzenje",
        "osnovati-fondaciju",
        "osnovati fondaciju"
    ]):
        return "procedura"

    if any(x in title_n or x in url_n for x in [
        "registracija-udruzenja",
        "registracija udruzenja",
        "registracija-fondacije",
        "registracija fondacije",
        "upis-u-registar",
        "upis u registar"
    ]):
        return "registracija"

    # tek na kraju zakon, jer se riječ zakon pojavljuje na mnogo stranica
    if any(x in title_n or x in url_n for x in [
        "zakon", "zakoni", "propis", "propisi", "pravilnik",
        "pravilnici", "podzakonski", "konvencije", "ugovori", "ustav"
    ]):
        return "zakon"

    return "ostalo"

def detect_semantic_topic(category, subsection, title, url):
    s = norm(
        str(category) + " " +
        str(subsection) + " " +
        str(title) + " " +
        str(url)
    )

    rules = {
        "udruzenja": [
            "udruzenj", "udruženj"
        ],
        "fondacije": [
            "fondacij"
        ],
        "pravosudni_ispit": [
            "pravosudni"
        ],
        "strucni_upravni_ispit": [
            "strucni-upravni", "strucni upravni", "stručni upravni",
            "srednju-i-visu-skolsku-spremu", "visoku-skolsku-spremu",
            "suis"
        ],
        "besplatna_pravna_pomoc": [
            "besplatna-pravna-pomoc", "besplatna pravna pomoc",
            "ured-za-pruzanje-besplatne-pravne-pomoci"
        ],
        "medjunarodna_pravna_pomoc": [
            "medunarodna-pravna-pomoc", "medjunarodna pravna pomoc",
            "medunarodna pravna pomoc"
        ],
        "alimentacije": [
            "alimentacij", "izdrzavanje", "izdržavanje"
        ],
        "otmica_djece": [
            "otmica", "otmice-djece", "vracanje-djeteta", "vidjanje-djeteta"
        ],
        "crkve": [
            "crkve", "vjerske-zajednice", "vjerske zajednice"
        ],
        "pravna_lica": [
            "pravna-lica", "pravna lica", "institucije-bosne-i-hercegovine"
        ],
        "strane_nvo": [
            "strane-nevladine", "predstavnistva-stranih",
            "predstavnistvo", "stranih-nevladinih", "strane nvo"
        ],
        "zakoni": [
            "zakoni", "propisi", "podzakonski", "ustav", "ugovori", "konvencije"
        ],
        "registri": [
            "registar", "registri", "izvod-iz-registra", "javnost-registra"
        ]
    }

    for topic, keywords in rules.items():
        if any(k in s for k in keywords):
            return topic

    return "general"

def detect_generic_penalty(title, url):
    s = norm(title + " " + url)

    generic_patterns = [
        "ministarstvo",
        "oblasti rada",
        "sektor",
        "ured",
        "interno",
        "organizacija",
        "nadleznosti",
        "početna",
        "pocetna"
    ]
    
    if any(x in s for x in generic_patterns):
        return 0.25

    return 0.0


chunks["page_type"] = chunks.apply(
    lambda r: detect_page_type(
        r["title"],
        r["source_url"],
        r["text"]
    ),
    axis=1
)

chunks["semantic_topic"] = chunks.apply(
    lambda r: detect_semantic_topic(
        r["category"],
        r["subsection"],
        r["title"],
        r["source_url"]
    ),
    axis=1
)

chunks["generic_penalty"] = chunks.apply(
    lambda r: detect_generic_penalty(
        r["title"],
        r["source_url"]
    ),
    axis=1
)

chunks.to_csv(DATA_DIR / "chunks_v5.csv", index=False)
records.to_csv(DATA_DIR / "records_v5.csv", index=False)

summary = {
    "page_types": chunks["page_type"].value_counts().to_dict(),
    "semantic_topics": chunks["semantic_topic"].value_counts().to_dict(),
}

with open(DATA_DIR / "metadata_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("DONE")
print(summary)