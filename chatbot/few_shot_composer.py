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


def clean_text(text):
    text = str(text)

    noise_patterns = [
        r"Hrvatski Bosanski Српски English BS",
        r"Ministarstvo pravde Bosne i Hercegovine",
        r"Ministarstvo Oblasti rada Zakoni/Ugovori.*?Početna \|",
        r"Oblasti rada \|.*?\|",
    ]

    for pattern in noise_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    menu_words = [
        "Ministarstvo", "Oblasti rada", "Zakoni/Ugovori",
        "Projekti/Strategije", "Publikacije/Priručnici",
        "Tenderi/Javni oglasi", "Info/Pristup informacijama",
        "Hrvatski", "Bosanski", "English", "Početna"
    ]

    for word in menu_words:
        text = text.replace(word, " ")

    text = re.sub(r"\s+", " ", text).strip()
    return text


FEW_SHOT_STYLES = {
    "procedure": {
        "intro": "Postupak se može sažeti ovako:",
        "format": "steps",
    },
    "requirements": {
        "intro": "Za ovaj postupak potrebno je pripremiti sljedeću dokumentaciju:",
        "format": "bullets",
    },
    "fee": {
        "intro": "Prema dostupnim informacijama, za ovaj postupak su relevantni sljedeći iznosi ili podaci o uplati:",
        "format": "money",
    },
    "contact": {
        "intro": "Za ovu oblast možeš koristiti sljedeće kontakt informacije:",
        "format": "contact",
    },
    "form": {
        "intro": "Za ovaj postupak relevantni su sljedeći obrasci ili zahtjevi:",
        "format": "bullets",
    },
    "definition": {
        "intro": "Prema dostupnim informacijama:",
        "format": "paragraph",
    },
    "date": {
        "intro": "Prema dostupnim informacijama o terminima ili rokovima:",
        "format": "bullets",
    },
    "general": {
        "intro": "Prema dostupnim informacijama:",
        "format": "bullets",
    },
}


def detect_goal(question):
    q = norm(question)

    if any(x in q for x in ["koliko", "kosta", "košta", "cijena", "taksa", "troskovi", "troškovi", "uplata", "racun", "račun"]):
        return "fee"

    if any(x in q for x in ["kontakt", "telefon", "email", "e-mail", "mail", "fax", "faks", "kome", "javim", "obratim"]):
        return "contact"

    if any(x in q for x in ["sta treba", "šta treba", "dokumenti", "dokumentacija", "papiri", "priloziti", "priložiti", "potrebno"]):
        return "requirements"

    if any(x in q for x in ["obrazac", "formular", "formulari", "prijava"]):
        return "form"

    if any(x in q for x in ["kako", "na koji nacin", "procedura", "postupak", "registrovati", "osnovati", "osnujem"]):
        return "procedure"

    if any(x in q for x in ["ko je", "sta je", "šta je", "definisi", "definiši", "znaci", "znači"]):
        return "definition"

    if any(x in q for x in ["rok", "kada", "kad", "termin", "datum"]):
        return "date"

    return "general"


def split_facts(text):
    text = clean_text(text)

    parts = re.split(
        r"(?<=[.!?])\s+|\n+|\s+(?=[a-z]\)\s)|\s+(?=\d+[\).]\s)|\s+[-•]\s+",
        text
    )

    facts = []

    phone_re = re.compile(r"(?:\+?\d{1,3}[\s/-]?)?(?:\d{2,3}[\s/-]?){2,5}\d{2,3}")
    email_re = re.compile(r"[\w\.-]+@[\w\.-]+")

    for part in parts:
        fact = normalize_fact(part)

        # Keep short but valuable contact facts (phones/emails).
        if len(fact) < 35 and not (phone_re.search(fact) or email_re.search(fact)):
            continue

        if len(fact) > 420:
            fact = fact[:420].rsplit(" ", 1)[0] + "."

        if is_noise(fact):
            continue

        facts.append(fact)

    return facts


def normalize_fact(text):
    text = str(text).strip()

    text = re.sub(r"^[a-z]\)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\d+[\).\s]+", "", text)
    text = re.sub(r"^[-•]\s*", "", text)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = text.strip(" -•|;\n\t")

    replacements = {
        "pristojba": "taksa",
        "pristojbe": "takse",
        "podnositelj": "podnosilac",
        "u składu": "u skladu",
        "tijela": "organa",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    if text:
        text = text[0].upper() + text[1:]

    # Remove broken leading fragments like "Stva," that can appear after aggressive splitting.
    if "," in text:
        first, rest = text.split(",", 1)
        first_clean = first.strip()
        if 1 <= len(first_clean) <= 4 and first_clean.lower() not in {"npr", "dr", "mr", "itd"}:
            text = rest.strip()
            if text:
                text = text[0].upper() + text[1:]

    return text


def is_noise(text):
    t = norm(text)

    noise = [
        "hrvatski bosanski",
        "english",
        "pocetna",
        "početna",
        "oblasti rada",
        "zakoni/ugovori",
        "projekti/strategije",
        "publikacije/prirucnici",
        "tenderi/javni oglasi",
    ]

    return any(x in t for x in noise)


def score_fact(question, fact, source, goal):
    q = norm(question)
    f = norm(fact)

    q_tokens = [t for t in re.findall(r"\w+", q) if len(t) >= 4]

    score = 0.0

    for token in q_tokens:
        if token in f:
            score += 1.0

    goal_keywords = {
        "procedure": ["postupak", "podnosi", "podnijeti", "zahtjev", "prijava", "registracija", "upis", "osniva"],
        "requirements": ["potrebno", "prilaže", "prilaze", "dokument", "obrazac", "statut", "odluka", "dokaz", "zapisnik"],
        "fee": ["km", "taksa", "takse", "uplata", "račun", "racun", "troš", "tros", "prihoda"],
        "contact": ["telefon", "email", "e-mail", "kontakt", "fax", "faks"],
        "form": ["obrazac", "formular", "zahtjev", "prijava"],
        "definition": ["je", "predstavlja", "znači", "znaci", "podrazumijeva"],
        "date": ["termin", "datum", "rok", "godine"],
        "general": [],
    }

    for kw in goal_keywords.get(goal, []):
        if kw in f:
            score += 2.0

    page_type = str(source.get("page_type", ""))

    GOAL_PAGE_TYPE_MAP = {
        "fee": {"taksa", "troskovi", "troškovi"},
        "contact": {"kontakt", "nadleznosti", "nadležnosti"},
        "requirements": {"dokumentacija", "uslovi", "uvjeti"},
        "form": {"obrazac", "obrasci", "formular", "formulari"},
        "procedure": {"prijava", "postupak", "registracija", "upis", "promjena", "brisanje"},
        "date": {"termini", "rok", "rokovi"},
        "definition": {"zakon", "propis", "pravilnik"},
        "general": set(),
    }

    if page_type == goal:
        score += 2.0
    elif page_type in GOAL_PAGE_TYPE_MAP.get(goal, set()):
        score += 1.5

    # Strong bonus for explicit contact signals in extracted facts.
    if goal == "contact":
        if re.search(r"[\w\.-]+@[\w\.-]+", fact):
            score += 2.5
        if re.search(r"(?:\+?\d{1,3}[\s/-]?)?(?:\d{2,3}[\s/-]?){2,5}\d{2,3}", fact):
            score += 2.0

    if page_type == "ostalo":
        score -= 0.6

    if str(source.get("semantic_topic", "")) == "general":
        score -= 0.4

    if len(fact) > 260:
        score -= 0.4

    if fact.count(",") > 8:
        score -= 0.3

    return score


def extract_relevant_facts(question, results, max_facts=6):
    goal = detect_goal(question)
    scored = []
    seen = set()

    for source_rank, source in enumerate(results[:4]):
        facts = split_facts(source.get("text", ""))

        for fact in facts:
            key = norm(fact)[:180]

            if key in seen:
                continue

            seen.add(key)

            score = score_fact(question, fact, source, goal)
            score += max(0, 1.4 - source_rank * 0.35)

            if score > 0:
                scored.append((score, fact, source))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:max_facts]


def extract_money(facts):
    lines = []

    for fact in facts:
        if re.search(r"\d+[.,]?\d*\s*KM", fact):
            lines.append(fact)

    return lines


def extract_contact(facts):
    text = " ".join(facts)

    phones = list(dict.fromkeys(re.findall(
        r"(?:\+387\s?)?(?:\d{2,3}[\s/-]?){2,5}\d{2,3}",
        text
    )))

    emails = list(dict.fromkeys(re.findall(r"[\w\.-]+@[\w\.-]+", text)))

    return phones[:4], emails[:3]


def remove_duplicate_lines(lines):
    seen = set()
    output = []

    for line in lines:
        key = norm(line)[:140]

        if key in seen:
            continue

        seen.add(key)
        output.append(line)

    return output


def compose_few_shot_answer(question, results, intent=None):
    goal = detect_goal(question)
    style = FEW_SHOT_STYLES.get(goal, FEW_SHOT_STYLES["general"])

    extracted = extract_relevant_facts(question, results)
    facts = [fact for _, fact, _ in extracted]
    facts = remove_duplicate_lines(facts)

    if not facts:
        return (
            "Pronašla sam relevantne izvore, ali u njima nema dovoljno jasno izdvojenih informacija "
            "da bih sigurno formulisala odgovor. Preporučujem da pogledaš izvore ispod."
        )

    intro = style["intro"]
    fmt = style["format"]

    if fmt == "contact":
        phones, emails = extract_contact(facts)

        details = []

        if phones:
            details.append("Telefon: " + ", ".join(phones))

        if emails:
            details.append("E-mail: " + ", ".join(emails))

        if details:
            return intro + "\n\n" + "\n".join(details)

        return intro + "\n\n" + "\n".join(f"- {f}" for f in facts[:3])

    if fmt == "money":
        money_lines = extract_money(facts)

        if money_lines:
            return intro + "\n\n" + "\n".join(f"- {line}" for line in money_lines[:5])

        return (
            "U dostupnim izvorima se spominju takse ili troškovi, ali nisam mogla sigurno izdvojiti tačan iznos. "
            "Najbolje je provjeriti zvanični izvor ispod."
        )

    if fmt == "paragraph":
        selected = facts[:2]
        return intro + "\n\n" + " ".join(selected)

    if fmt == "steps":
        selected = facts[:5]
        answer = intro + "\n\n"
        for i, fact in enumerate(selected, start=1):
            answer += f"{i}. {fact}\n"
        return answer.strip()

    selected = facts[:5]
    return intro + "\n\n" + "\n".join(f"- {fact}" for fact in selected)


def answer_quality_bad(answer):
    a = norm(answer)

    bad = [
        "hrvatski bosanski",
        "oblasti rada",
        "zakoni/ugovori",
        "projekti/strategije",
        "publikacije/prirucnici",
    ]

    if any(x in a for x in bad):
        return True

    if len(str(answer).strip()) < 45:
        return True

    return False