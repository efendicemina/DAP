import re


def norm(text):
    text = str(text).lower().strip()
    text = (
        text.replace("č", "c")
        .replace("ć", "c")
        .replace("š", "s")
        .replace("ž", "z")
        .replace("đ", "dj")
    )
    return re.sub(r"\s+", " ", text)


ANSWER_CARDS = [
    {
        "id": "osnivanje_udruzenja",
        "intent": "registracija",
        "keywords_any": ["udruzenj"],
        "keywords_all_any": [["osnov", "naprav", "formir", "registrov"]],
        "title": "Kako osnovati udruženje",
        "answer": (
            "Udruženje se može osnovati ako postoje najmanje tri osnivača. "
            "Osnivači mogu biti fizička lica, pravna lica ili kombinacija fizičkih i pravnih lica, "
            "uz uslov da broj osnivača ne bude manji od tri.\n\n"
            "Za osnivanje udruženja potrebno je da osnivačka skupština donese osnivački akt, statut "
            "i odluku o imenovanju organa upravljanja. Nakon toga se priprema zahtjev i prateća dokumentacija "
            "za upis u registar kod Ministarstva pravde BiH.\n\n"
            "U praksi, prvi korak je priprema osnovnih akata udruženja, a zatim podnošenje dokumentacije Ministarstvu."
        ),
        "sources": [
            {
                "title": "Kako osnovati udruženje?",
                "url": "https://mpr.gov.ba/bs/kako-osnovati-udruzenje"
            },
            {
                "title": "Potrebna dokumentacija",
                "url": "https://mpr.gov.ba/bs/potrebna-dokumentacija"
            }
        ]
    },
    {
        "id": "dokumentacija_udruzenja",
        "intent": "registracija",
        "keywords_any": ["udruzenj"],
        "keywords_all_any": [["dokument", "papir", "sta treba", "šta treba", "potrebno"]],
        "title": "Dokumentacija za registraciju udruženja",
        "answer": (
            "Za registraciju udruženja potrebno je pripremiti zahtjev za upis u registar i prateću dokumentaciju. "
            "Najvažniji dokumenti su Obrazac 1, spisak osnivača, zapisnik sa osnivačke skupštine, osnivački akt, "
            "statut u dva primjerka, odluka o imenovanju organa udruženja, ovjeren potpis osobe ovlaštene za zastupanje "
            "i dokaz o uplati administrativne takse.\n\n"
            "Ako udruženje ima naziv na stranom jeziku ili koristi ime fizičke osobe, mogu biti potrebne i dodatne saglasnosti "
            "ili ovjereni prevodi."
        ),
        "sources": [
            {
                "title": "Potrebna dokumentacija",
                "url": "https://mpr.gov.ba/bs/potrebna-dokumentacija"
            }
        ]
    },
    {
        "id": "taksa_brisanje_udruzenja",
        "intent": "registracija",
        "keywords_any": ["udruzenj"],
        "keywords_all_any": [["bris", "izbris", "ugas"], ["taksa", "kosta", "košta", "cijena"]],
        "title": "Taksa za brisanje udruženja",
        "answer": (
            "Za zahtjev za upis brisanja iz registra udruženja, fondacija i drugih neprofitnih organizacija "
            "navedena je administrativna taksa od 10,00 KM.\n\n"
            "Uplata se vrši na depozitni račun Ministarstva finansija i trezora BiH, uz odgovarajuću vrstu prihoda "
            "i budžetsku organizaciju navedenu u izvoru."
        ),
        "sources": [
            {
                "title": "Administrativna taksa",
                "url": "https://mpr.gov.ba/bs/administrativna-taksa84"
            }
        ]
    },
    {
        "id": "taksa_registracija_udruzenja",
        "intent": "registracija",
        "keywords_any": ["udruzenj"],
        "keywords_all_any": [["registracij", "upis"], ["taksa", "kosta", "košta", "cijena"]],
        "title": "Taksa za registraciju udruženja",
        "answer": (
            "Za zahtjev za upis u registar udruženja navedena je administrativna taksa od 200,00 KM.\n\n"
            "Uz zahtjev je potrebno dostaviti dokaz o uplati administrativne takse. Podaci o računima i svrsi uplate "
            "navedeni su na stranici Ministarstva."
        ),
        "sources": [
            {
                "title": "Administrativna taksa",
                "url": "https://mpr.gov.ba/bs/administrativna-taksa55"
            }
        ]
    },
    {
        "id": "promjena_podataka_udruzenje",
        "intent": "registracija",
        "keywords_any": ["udruzenj", "organizacij"],
        "keywords_all_any": [["promjen", "izmjen", "preregistr", "mandat", "zastupnik", "adresa"]],
        "title": "Promjena podataka u registru",
        "answer": (
            "Za promjenu podataka o udruženju potrebno je Ministarstvu podnijeti zahtjev za upis promjena u registar. "
            "Uz zahtjev se obično prilaže dokumentacija koja dokazuje promjenu, na primjer odluka nadležnog organa, "
            "izmjene statuta, podaci o novom zastupniku ili drugi relevantni akti.\n\n"
            "Ako se promjena odnosi na mandat, zastupnika, adresu ili naziv, najbolje je provjeriti odgovarajuće formulare "
            "i uputstvo na stranici Ministarstva."
        ),
        "sources": [
            {
                "title": "Upis promjena u Registru udruženja ili fondacija",
                "url": "https://mpr.gov.ba/bs/upis-promjena-u-registru-udruzenja-ili-fondacija"
            },
            {
                "title": "Formulari",
                "url": "https://mpr.gov.ba/bs/formulari97"
            }
        ]
    },
    {
        "id": "osnivanje_fondacije",
        "intent": "registracija",
        "keywords_any": ["fondacij"],
        "keywords_all_any": [["osnov", "naprav", "formir", "registrov", "sta treba", "šta treba"]],
        "title": "Osnivanje fondacije",
        "answer": (
            "Za osnivanje fondacije potrebno je pripremiti dokumentaciju za registraciju fondacije pri Ministarstvu pravde BiH. "
            "Postupak obuhvata pripremu osnivačkog akta, statuta i drugih dokumenata propisanih za upis fondacije u registar.\n\n"
            "Za konkretan spisak dokumenata i obrazaca potrebno je koristiti stranice Ministarstva koje se odnose na registraciju fondacije."
        ),
        "sources": [
            {
                "title": "Kako osnovati fondaciju",
                "url": "https://mpr.gov.ba/bs/kako-osnovati-fondaciju"
            },
            {
                "title": "Registracija fondacije",
                "url": "https://mpr.gov.ba/bs/registracija-fondacije"
            }
        ]
    },
    {
        "id": "pravosudni_ispit_prijava",
        "intent": "ispiti",
        "keywords_any": ["pravosudni"],
        "keywords_all_any": [["prijav", "podnijeti", "podnosenje", "podnošenje"]],
        "title": "Prijava za pravosudni ispit",
        "answer": (
            "Za prijavu na pravosudni ispit potrebno je pratiti uputstva Ministarstva pravde BiH za podnošenje prijave. "
            "Na stranici za prijavu nalaze se informacije o načinu podnošenja zahtjeva i dokumentaciji koja se prilaže.\n\n"
            "Ako korisnik nije siguran da li ispunjava uslove, prvo treba provjeriti stranicu sa uslovima za polaganje pravosudnog ispita."
        ),
        "sources": [
            {
                "title": "Prijava polaganja pravosudnog ispita",
                "url": "https://mpr.gov.ba/bs/prijava-polaganja-pravosudnog-ispita"
            },
            {
                "title": "Uslovi za polaganje ispita",
                "url": "https://mpr.gov.ba/bs/uslovi-za-polaganje-ispita-"
            }
        ]
    },
    {
        "id": "strucni_upravni_kontakt",
        "intent": "kontakt_i_nadleznosti",
        "keywords_any": ["strucni upravni", "stručni upravni"],
        "keywords_all_any": [["kontakt", "telefon", "email", "mail", "javim"]],
        "title": "Kontakt za stručni upravni ispit",
        "answer": (
            "Za informacije o stručnom upravnom ispitu potrebno je kontaktirati nadležnu službu Ministarstva pravde BiH. "
            "Kontakt podaci se nalaze na stranici Ministarstva za stručni upravni ispit.\n\n"
            "Ako se pitanje odnosi na visoku školsku spremu, koristi se kontakt za taj dio ispita; ako se odnosi na srednju ili višu stručnu spremu, "
            "potrebno je provjeriti odgovarajuću kontakt stranicu."
        ),
        "sources": [
            {
                "title": "Kontakt",
                "url": "https://mpr.gov.ba/bs/kontakt12"
            },
            {
                "title": "Kontakt",
                "url": "https://mpr.gov.ba/bs/kontakt13"
            }
        ]
    },
    {
        "id": "besplatna_pravna_pomoc",
        "intent": "pravna_pomoc",
        "keywords_any": ["pravna pomoc", "pravnu pomoc", "besplatna"],
        "keywords_all_any": [["pravna", "pomoc"]],
        "title": "Besplatna pravna pomoć",
        "answer": (
            "Besplatna pravna pomoć se odnosi na pružanje pravne podrške osobama koje ostvaruju ili štite svoja prava i zakonom zaštićene interese. "
            "Za tačne uslove, način podnošenja zahtjeva i nadležni kontakt potrebno je provjeriti stranicu Ureda za pružanje besplatne pravne pomoći.\n\n"
            "Ako je pitanje vezano za konkretan postupak, preporučuje se da korisnik kontaktira nadležni ured i provjeri koju dokumentaciju treba dostaviti."
        ),
        "sources": [
            {
                "title": "Besplatna pravna pomoć",
                "url": "https://mpr.gov.ba/bs/besplatna-pravna-pomoc"
            },
            {
                "title": "Ured za pružanje besplatne pravne pomoći",
                "url": "https://mpr.gov.ba/bs/ured-za-pruzanje-besplatne-pravne-pomoci"
            }
        ]
    },
    {
        "id": "medjunarodna_otmica_djece",
        "intent": "pravna_pomoc",
        "keywords_any": ["otmica", "dijete", "djeteta", "djece"],
        "keywords_all_any": [["otmic", "vracanje", "viđanje", "vidjanje", "dijete", "djeteta"]],
        "title": "Međunarodna otmica djece",
        "answer": (
            "Ako se pitanje odnosi na međunarodnu otmicu djeteta, potrebno je postupati hitno i obratiti se nadležnim institucijama. "
            "Ministarstvo pravde BiH ima informacije o postupanju u slučaju otmice djece, zahtjevima i obrascima za vraćanje ili viđanje djeteta.\n\n"
            "U hitnim situacijama, posebno ako je dijete u neposrednoj opasnosti, potrebno je odmah kontaktirati policiju."
        ),
        "sources": [
            {
                "title": "Postupanje u slučaju otmice djece",
                "url": "https://mpr.gov.ba/bs/postupanje-u-slucaju-otmice-djece"
            },
            {
                "title": "Obrazac zahtjeva za vraćanje djeteta",
                "url": "https://mpr.gov.ba/bs/obrazac-zahtjeva-za-vracanje-djeteta"
            }
        ]
    },
]


def match_answer_card(question, intent=None):
    q = norm(question)

    best = None
    best_score = 0

    for card in ANSWER_CARDS:
        score = 0

        if intent and card.get("intent") == intent:
            score += 2

        for kw in card.get("keywords_any", []):
            if norm(kw) in q:
                score += 2

        groups = card.get("keywords_all_any", [])
        for group in groups:
            if any(norm(kw) in q for kw in group):
                score += 2

        if score > best_score:
            best = card
            best_score = score

    if best_score >= 4:
        return best

    return None