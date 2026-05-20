import json
import random
import re
from pathlib import Path
from collections import Counter

import pandas as pd

# =========================
# CONFIG
# =========================

OUTPUT_DIR = Path("mpr_dataset_final")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TARGET_SIZE = 5000
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

ADD_TYPOS_PROB = 0.10
ADD_INFORMAL_PROB = 0.25
ADD_CONTEXT_PROB = 0.20

# =========================
# INTENT SCHEMA
# =========================

INTENT_SCHEMA = {
    "registracija": {
        "subintents": {
            "registracija_udruzenja": [
                "Kako registrovati udruženje?",
                "Šta mi treba za registraciju udruženja?",
                "Koja dokumentacija je potrebna za osnivanje udruženja?",
                "Gdje se predaje zahtjev za upis udruženja?",
                "Kako ide postupak upisa udruženja u registar?"
            ],
            "registracija_fondacije": [
                "Kako registrovati fondaciju?",
                "Koji su uslovi za osnivanje fondacije?",
                "Šta treba predati za registraciju fondacije?",
                "Gdje se podnosi zahtjev za registraciju fondacije?",
                "Koja je procedura za upis fondacije?"
            ],
            "promjena_podataka": [
                "Kako prijaviti promjenu podataka udruženja?",
                "Šta treba za promjenu osobe ovlaštene za zastupanje?",
                "Kako promijeniti adresu udruženja?",
                "Koja dokumentacija treba za izmjenu podataka u registru?",
                "Kako se radi preregistracija udruženja?"
            ],
            "brisanje_iz_registra": [
                "Kako se briše udruženje iz registra?",
                "Šta treba za prestanak rada udruženja?",
                "Kako podnijeti zahtjev za brisanje fondacije?",
                "Koji dokumenti trebaju za likvidaciju udruženja?",
                "Kako odjaviti udruženje iz registra?"
            ]
        }
    },

    "obrasci": {
        "subintents": {
            "obrazac_registracija": [
                "Gdje mogu preuzeti obrazac za registraciju udruženja?",
                "Treba mi formular za registraciju fondacije.",
                "Koji obrazac se koristi za osnivanje udruženja?",
                "Ima li zahtjev za registraciju u Word ili PDF formatu?",
                "Gdje je prijavni obrazac za upis u registar?"
            ],
            "obrazac_promjena": [
                "Treba mi obrazac za promjenu podataka udruženja.",
                "Gdje mogu naći zahtjev za izmjenu podataka?",
                "Koji formular treba za promjenu zastupnika?",
                "Ima li obrazac za preregistraciju?",
                "Kako se zove obrazac za izmjenu u registru?"
            ],
            "obrazac_ispiti": [
                "Gdje je obrazac za prijavu pravosudnog ispita?",
                "Treba mi prijava za stručni upravni ispit.",
                "Koji formular se predaje za polaganje ispita?",
                "Ima li obrazac za popravni ispit?",
                "Gdje se preuzima prijava za ispit?"
            ],
            "obrazac_zospi": [
                "Gdje je obrazac za pristup informacijama?",
                "Kako podnijeti zahtjev po ZOSPI-u?",
                "Treba mi formular za slobodu pristupa informacijama.",
                "Ima li zahtjev za pristup informacijama?",
                "Koji obrazac ide za ZOSPI?"
            ]
        }
    },

    "zakoni_i_propisi": {
        "subintents": {
            "zakoni_udruzenja": [
                "Koji zakon reguliše udruženja i fondacije?",
                "Gdje mogu naći zakon o udruženjima?",
                "Koji propisi važe za fondacije?",
                "Treba mi pravni osnov za registraciju udruženja.",
                "Koji zakon se primjenjuje na NVO?"
            ],
            "zakoni_ispiti": [
                "Koji zakon reguliše pravosudni ispit?",
                "Gdje su propisi za stručni upravni ispit?",
                "Treba mi pravilnik o polaganju ispita.",
                "Koji zakon propisuje uslove za pravosudni ispit?",
                "Gdje mogu naći program pravosudnog ispita?"
            ],
            "zakoni_zospi": [
                "Gdje je zakon o slobodi pristupa informacijama?",
                "Koji propis reguliše pristup informacijama?",
                "Treba mi pravilnik za ZOSPI.",
                "Kako je regulisan zahtjev za pristup informacijama?",
                "Gdje se nalazi vodič za pristup informacijama?"
            ],
            "zakoni_notari_tumaci": [
                "Koji propisi važe za notare?",
                "Gdje su pravilnici za sudske tumače?",
                "Koji zakon reguliše notarsku službu?",
                "Treba mi propis o stalnim sudskim tumačima.",
                "Gdje mogu naći pravila za imenovanje sudskog tumača?"
            ]
        }
    },

    "pravna_pomoc": {
        "subintents": {
            "besplatna_pravna_pomoc": [
                "Kako mogu dobiti besplatnu pravnu pomoć?",
                "Ko ima pravo na besplatnu pravnu pomoć?",
                "Gdje se predaje zahtjev za pravnu pomoć?",
                "Koji dokumenti trebaju za besplatnu pravnu pomoć?",
                "Kome se obratiti za pravnu pomoć?"
            ],
            "medjunarodna_pravna_pomoc": [
                "Kako funkcioniše međunarodna pravna pomoć?",
                "Kome se obratiti za međunarodnu pravnu pomoć?",
                "Treba mi pomoć za predmet u inostranstvu.",
                "Kako ide pravna saradnja sa drugom državom?",
                "Ko je nadležan za međunarodnu pravnu pomoć?"
            ],
            "alimentacija": [
                "Kako naplatiti alimentaciju iz inostranstva?",
                "Kome se obratiti za alimentaciju ako je roditelj van BiH?",
                "Koji dokumenti trebaju za međunarodnu alimentaciju?",
                "Kako pokrenuti postupak za izdržavanje iz druge države?",
                "Treba mi pomoć za alimentaciju iz inostranstva."
            ],
            "otmica_djece": [
                "Šta uraditi u slučaju međunarodne otmice djeteta?",
                "Kome prijaviti odvođenje djeteta u drugu državu?",
                "Kako se pokreće postupak za povrat djeteta?",
                "Ko je nadležan za otmicu djece preko granice?",
                "Treba mi informacija o Haškoj konvenciji za djecu."
            ]
        }
    },

    "ispiti": {
        "subintents": {
            "pravosudni_ispit": [
                "Ko može polagati pravosudni ispit?",
                "Kako se prijavljuje pravosudni ispit?",
                "Koji su uslovi za pravosudni ispit?",
                "Koliko košta polaganje pravosudnog ispita?",
                "Kada su termini za pravosudni ispit?"
            ],
            "strucni_upravni_ispit": [
                "Ko može polagati stručni upravni ispit?",
                "Kako se prijavljuje stručni upravni ispit?",
                "Koji dokumenti trebaju za stručni upravni ispit?",
                "Kada su termini za stručni upravni ispit?",
                "Gdje se polaže stručni upravni ispit?"
            ],
            "kontakt_ispiti": [
                "Koga mogu kontaktirati za pravosudni ispit?",
                "Koji je email za stručni upravni ispit?",
                "Gdje da pitam za ispitne termine?",
                "Treba mi broj telefona za informacije o ispitu.",
                "Kome se šalje prijava za ispit?"
            ],
            "termini_ispita": [
                "Kada je sljedeći termin ispita?",
                "Ima li raspored za polaganje ispita?",
                "Gdje mogu vidjeti listu kandidata za ispit?",
                "Kada je popravni ispit?",
                "Jesu li objavljeni novi ispitni termini?"
            ]
        }
    },

    "registri": {
        "subintents": {
            "registar_udruzenja": [
                "Gdje mogu pronaći registar udruženja?",
                "Kako provjeriti da li je udruženje registrovano?",
                "Treba mi izvod iz registra udruženja.",
                "Kako pronaći podatke o udruženju?",
                "Gdje je javni registar fondacija?"
            ],
            "registar_notara": [
                "Gdje mogu pronaći listu notara?",
                "Kako provjeriti notara?",
                "Ima li registar notara?",
                "Treba mi spisak notara u BiH.",
                "Gdje su objavljeni notari?"
            ],
            "registar_tumaca": [
                "Gdje mogu pronaći sudskog tumača?",
                "Ima li lista stalnih sudskih tumača?",
                "Kako provjeriti da li je neko sudski tumač?",
                "Treba mi registar sudskih tumača.",
                "Gdje su objavljeni ovlašteni sudski tumači?"
            ]
        }
    },

    "kontakt_i_nadleznosti": {
        "subintents": {
            "kontakt": [
                "Kako mogu kontaktirati Ministarstvo pravde?",
                "Koji je email Ministarstva pravde BiH?",
                "Gdje se nalazi Ministarstvo pravde?",
                "Koji je broj telefona Ministarstva?",
                "Kako poslati upit Ministarstvu?"
            ],
            "nadleznosti": [
                "Šta je nadležnost Ministarstva pravde BiH?",
                "Za šta je nadležno Ministarstvo pravde?",
                "Koji sektor radi registraciju udruženja?",
                "Kome se obratiti za pravnu pomoć?",
                "Ko je nadležan za pravosudni ispit?"
            ],
            "organizacija": [
                "Gdje mogu vidjeti organizaciju Ministarstva?",
                "Koji sektori postoje u Ministarstvu?",
                "Kako pronaći kontakt osobe u Ministarstvu?",
                "Treba mi organizaciona struktura Ministarstva.",
                "Ko radi u sektoru za pravosuđe?"
            ]
        }
    },

    "out_of_scope": {
        "subintents": {
            "nevezano": [
                "Kako da produžim pasoš?",
                "Gdje se vadi lična karta?",
                "Kako registrovati auto?",
                "Kada rade općinske službe?",
                "Kako dobiti uvjerenje o državljanstvu?",
                "Gdje se plaća parking kazna?",
                "Kako otvoriti firmu u poreznoj upravi?",
                "Koji su uslovi za vozačku dozvolu?",
                "Kako se prijaviti na biro za zapošljavanje?",
                "Gdje se predaje zahtjev za građevinsku dozvolu?"
            ]
        }
    }
}

# =========================
# VARIATION BANKS
# =========================

PREFIXES = [
    "", "", "", "",
    "Pozdrav, ",
    "Dobar dan, ",
    "Molim vas, ",
    "Može pomoć, ",
    "Trebam informaciju, ",
    "Zanima me, ",
    "Ne mogu da nađem na stranici, ",
]

SUFFIXES = [
    "", "", "", "",
    " Hvala.",
    " Ako može.",
    " Molim odgovor.",
    " Treba mi hitno.",
    " Nisam siguran gdje da tražim.",
    " Možete li me uputiti?",
]

SHORT_FORMS = [
    "pravosudni ispit prijava",
    "stručni upravni ispit termini",
    "obrazac registracija udruženja",
    "zakon udruženja fondacije",
    "kontakt ministarstvo pravde",
    "registar sudskih tumača",
    "lista notara",
    "besplatna pravna pomoć zahtjev",
    "zospi obrazac",
    "promjena podataka udruženje"
]

TYPO_MAP = {
    "č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "dj",
    "ije": "je",
    "pravosudni": "pravosudni",
    "stručni": "strucni",
    "udruženje": "udruzenje",
    "fondacija": "fondacija",
    "obrazac": "obrazac",
    "registracija": "registracija",
}

def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()

def maybe_add_typo(text):
    if random.random() > ADD_TYPOS_PROB:
        return text

    result = text

    replacements = [
        ("č", "c"), ("ć", "c"), ("š", "s"), ("ž", "z"), ("đ", "dj"),
        ("Ministarstvo", "ministarstvo"),
        ("pravosudni", "pravosudni"),
        ("stručni", "strucni"),
        ("udruženja", "udruzenja"),
        ("pomoć", "pomoc"),
    ]

    random.shuffle(replacements)

    for old, new in replacements[: random.randint(1, 3)]:
        result = result.replace(old, new)

    return result

def maybe_make_informal(text):
    if random.random() > ADD_INFORMAL_PROB:
        return text

    informal_replacements = [
        ("Kako mogu", "Kako da"),
        ("Gdje mogu pronaći", "Gdje da nađem"),
        ("Koji dokumenti trebaju", "Šta mi treba"),
        ("Ko može", "Ko smije"),
        ("Kome se obratiti", "Koga da kontaktiram"),
        ("podnosi", "predaje"),
        ("zahtjev", "zahtjev/prijava"),
    ]

    result = text

    for old, new in informal_replacements:
        if old in result and random.random() < 0.5:
            result = result.replace(old, new)

    return result

def maybe_add_context(text):
    if random.random() > ADD_CONTEXT_PROB:
        return text

    contexts = [
        "Radim ovo prvi put. ",
        "Ne razumijem proceduru. ",
        "Treba mi za fakultetski zadatak. ",
        "Ne znam koji dokument da predam. ",
        "Na stranici ima puno linkova. ",
        "Pokušavam naći tačan obrazac. ",
    ]

    return random.choice(contexts) + text

def augment_question(question):
    question = maybe_make_informal(question)
    question = maybe_add_context(question)
    question = random.choice(PREFIXES) + question + random.choice(SUFFIXES)
    question = maybe_add_typo(question)
    return normalize_spaces(question)

def make_short_query(intent):
    if intent == "out_of_scope":
        return random.choice([
            "lična karta", "pasoš", "vozačka dozvola", "parking kazna",
            "građevinska dozvola", "porezna firma", "biro prijava"
        ])

    return random.choice(SHORT_FORMS)

# =========================
# GENERATION
# =========================

def flatten_seed_questions():
    rows = []

    for intent, intent_data in INTENT_SCHEMA.items():
        for subintent, questions in intent_data["subintents"].items():
            for q in questions:
                rows.append({
                    "intent": intent,
                    "subintent": subintent,
                    "canonical_question": q
                })

    return rows

def generate_dataset(target_size=TARGET_SIZE):
    seed_rows = flatten_seed_questions()
    intents = list(INTENT_SCHEMA.keys())

    # balanced, but out_of_scope intentionally smaller
    weights = {
        "registracija": 0.13,
        "obrasci": 0.14,
        "zakoni_i_propisi": 0.14,
        "pravna_pomoc": 0.12,
        "ispiti": 0.14,
        "registri": 0.11,
        "kontakt_i_nadleznosti": 0.10,
        "out_of_scope": 0.12,
    }

    target_per_intent = {
        intent: int(target_size * weights[intent])
        for intent in intents
    }

    # fix rounding
    while sum(target_per_intent.values()) < target_size:
        target_per_intent[random.choice(intents)] += 1

    rows = []
    seen = set()

    for intent in intents:
        intent_seed = [r for r in seed_rows if r["intent"] == intent]
        needed = target_per_intent[intent]

        attempts = 0

        while len([r for r in rows if r["label"] == intent]) < needed:
            attempts += 1

            if attempts > needed * 100:
                raise RuntimeError(f"Could not generate enough unique examples for {intent}")

            seed = random.choice(intent_seed)

            if random.random() < 0.15:
                text = make_short_query(intent)
                difficulty = "short_query"
            else:
                text = augment_question(seed["canonical_question"])
                difficulty = random.choice(["easy", "medium", "natural"])

            key = normalize_spaces(text.lower())

            if key in seen:
                continue

            seen.add(key)

            rows.append({
                "text": text,
                "label": intent,
                "subintent": seed["subintent"],
                "canonical_question": seed["canonical_question"],
                "difficulty": difficulty,
                "source": "synthetic_rule_based"
            })

    random.shuffle(rows)
    return rows

def assign_splits(rows):
    by_label = {}

    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    final_rows = []

    for label, items in by_label.items():
        random.shuffle(items)

        n = len(items)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        for i, row in enumerate(items):
            if i < train_end:
                split = "train"
            elif i < val_end:
                split = "val"
            else:
                split = "test"

            row["split"] = split
            final_rows.append(row)

    random.shuffle(final_rows)
    return final_rows

def save_outputs(rows):
    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_DIR / "intent_dataset_v2.csv", index=False)

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        split_df.to_csv(OUTPUT_DIR / f"intent_{split}.csv", index=False)

    with open(OUTPUT_DIR / "intent_dataset_v2.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total_rows": len(df),
        "labels": df["label"].value_counts().to_dict(),
        "subintents": df["subintent"].value_counts().to_dict(),
        "splits": df["split"].value_counts().to_dict(),
        "difficulty": df["difficulty"].value_counts().to_dict(),
    }

    with open(OUTPUT_DIR / "intent_dataset_v2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("DONE")
    print("Saved to:", OUTPUT_DIR)
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print("\nSplit distribution:")
    print(df["split"].value_counts())
    print("\nExamples:")
    print(df.sample(15, random_state=RANDOM_SEED)[["text", "label", "subintent", "split"]])

def main():
    rows = generate_dataset(TARGET_SIZE)
    rows = assign_splits(rows)
    save_outputs(rows)

if __name__ == "__main__":
    main()