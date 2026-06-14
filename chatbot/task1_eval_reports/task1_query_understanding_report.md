# Task 1 evaluacija: analiza korisničkog upita

## Cilj taska

Cilj prvog NLP taska je razumjeti korisnički upit prije retrieval i chatbot faze. Task se sastoji od dvije komponente:

1. **Intent classification** - određivanje glavne namjere korisnika.
2. **Ekstrakcija ključnih informacija** - izdvajanje tipa tražene informacije, dokumenta, procedure, pravne oblasti, institucije i ključnih riječi.

## Dvije forme prikaza teksta

| Pristup | Forma prikaza teksta | Opis |
|---|---|---|
| Baseline | TF-IDF n-gram reprezentacija | Klasična leksička reprezentacija korisničkog pitanja. |
| Kontekstualni model | SentenceTransformer embeddings | Duboko kontekstualna reprezentacija koja bolje hvata parafraze i semantički slične formulacije. |

## Rezultati intent klasifikacije

| Model | Split | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| TF-IDF + Logistic Regression | validation | 0.9980 | 0.9981 | 0.9980 |
| TF-IDF + Logistic Regression | test | 0.9820 | 0.9798 | 0.9819 |
| SentenceTransformer + Logistic Regression | validation | 0.9740 | 0.9750 | 0.9741 |
| SentenceTransformer + Logistic Regression | test | 0.9600 | 0.9587 | 0.9598 |

## Rezultati ekstrakcije ključnih informacija

Ekstrakcija procedura je evaluirana pomoću weak-label pristupa, gdje se očekivana procedura izvodi iz `subintent` kolone sintetičkog skupa podataka.

| Metrika | Vrijednost |
|---|---:|
| Evaluirani primjeri | 369 |
| Tačni primjeri | 254 |
| Preskočeni bez weak labela | 131 |
| Procedure extraction accuracy | 0.6883 |

## Primjeri strukturiranog izlaza

```json
[
  {
    "original_query": "Pozdrav, Gdje se vadi lična karta? Nisam siguran gdje da tražim.",
    "normalized_query": "pozdrav gdje se vadi licna karta nisam siguran gdje da trazim",
    "action_type": "opce_pitanje",
    "document_types": [],
    "procedures": [],
    "legal_topics": [],
    "institutions": [],
    "keywords": [
      "pozdrav",
      "vadi",
      "licna",
      "karta",
      "nisam",
      "siguran",
      "trazim"
    ]
  },
  {
    "original_query": "Zanima me, Pokušavam naći tačan obrazac. Koja dokumentacija je potrebna za osnivanje udruženja?",
    "normalized_query": "zanima me pokusavam naci tacan obrazac koja dokumentacija je potrebna za osnivanje udruzenja",
    "action_type": "obrazac",
    "document_types": [
      "obrazac"
    ],
    "procedures": [
      "registracija_udruzenja"
    ],
    "legal_topics": [
      "udruzenja_i_fondacije"
    ],
    "institutions": [],
    "keywords": [
      "zanima",
      "pokusavam",
      "naci",
      "tacan",
      "obrazac",
      "dokumentacija",
      "potrebna",
      "osnivanje"
    ]
  },
  {
    "original_query": "Pozdrav, Koji propisi važe za fondacije? Nisam siguran gdje da tražim.",
    "normalized_query": "pozdrav koji propisi vaze za fondacije nisam siguran gdje da trazim",
    "action_type": "pravni_akt",
    "document_types": [],
    "procedures": [],
    "legal_topics": [
      "udruzenja_i_fondacije",
      "zakoni_i_propisi"
    ],
    "institutions": [],
    "keywords": [
      "pozdrav",
      "propisi",
      "vaze",
      "fondacije",
      "nisam",
      "siguran",
      "trazim"
    ]
  },
  {
    "original_query": "Molim vas, Kako dobiti uvjerenje o drzavljanstvu?",
    "normalized_query": "molim vas kako dobiti uvjerenje o drzavljanstvu",
    "action_type": "procedura",
    "document_types": [],
    "procedures": [],
    "legal_topics": [],
    "institutions": [],
    "keywords": [
      "molim",
      "vas",
      "dobiti",
      "uvjerenje",
      "drzavljanstvu"
    ]
  },
  {
    "original_query": "Zanima me, Koji propisi važe za notare? Ako može.",
    "normalized_query": "zanima me koji propisi vaze za notare ako moze",
    "action_type": "pravni_akt",
    "document_types": [],
    "procedures": [],
    "legal_topics": [
      "registri",
      "zakoni_i_propisi"
    ],
    "institutions": [
      "notar"
    ],
    "keywords": [
      "zanima",
      "propisi",
      "vaze",
      "notare"
    ]
  },
  {
    "original_query": "Dobar dan, Gdje je obrazac za pristup informacijama? Treba mi hitno.",
    "normalized_query": "dobar dan gdje je obrazac za pristup informacijama treba mi hitno",
    "action_type": "obrazac",
    "document_types": [
      "obrazac"
    ],
    "procedures": [
      "zospi"
    ],
    "legal_topics": [],
    "institutions": [],
    "keywords": [
      "dobar",
      "dan",
      "obrazac",
      "pristup",
      "informacijama",
      "hitno"
    ]
  },
  {
    "original_query": "Pokušavam naći tačan obrazac. Gdje mogu pronaći registar udruženja? Hvala.",
    "normalized_query": "pokusavam naci tacan obrazac gdje mogu pronaci registar udruzenja hvala",
    "action_type": "obrazac",
    "document_types": [
      "obrazac",
      "registar"
    ],
    "procedures": [
      "registar_udruzenja"
    ],
    "legal_topics": [
      "udruzenja_i_fondacije",
      "registri"
    ],
    "institutions": [],
    "keywords": [
      "pokusavam",
      "naci",
      "tacan",
      "obrazac",
      "pronaci",
      "registar",
      "udruzenja",
      "hvala"
    ]
  },
  {
    "original_query": "Može pomoć, Kako dobiti uvjerenje o državljanstvu? Nisam siguran gdje da tražim.",
    "normalized_query": "moze pomoc kako dobiti uvjerenje o drzavljanstvu nisam siguran gdje da trazim",
    "action_type": "procedura",
    "document_types": [],
    "procedures": [],
    "legal_topics": [],
    "institutions": [],
    "keywords": [
      "pomoc",
      "dobiti",
      "uvjerenje",
      "drzavljanstvu",
      "nisam",
      "siguran",
      "trazim"
    ]
  }
]
```

## Zaključak

Task 1 sada formalno ispunjava zahtjev projekta jer ima dvije različite forme prikaza teksta, pri čemu je druga duboko kontekstualna. Pored same klasifikacije namjere, dodat je i interpretabilan sloj za ekstrakciju ključnih informacija iz korisničkog pitanja.
