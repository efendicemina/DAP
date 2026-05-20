"""
Dummy response generator for chatbot during development.
Will be replaced with actual ML model later.
"""
import random
from typing import Dict, List

# Dummy knowledge base for testing
DUMMY_KB = {
    "registracija": {
        "responses": [
            "Za registraciju udruženja građana potrebni su: statut, seznam članova, dokaz o identitetu. Pogledajte obrazac na https://www.mpr.gov.ba/bs/obrasci.",
            "Registracija traje obično 15-30 dana. Podnijite zahtjev na lokalnoj poslovnici ili putem eGoverna.",
            "Zakoni koji reguliše registraciju: Zakon o udrugama, Zakon o fondacijama. Detaljno na https://www.mpr.gov.ba/bs/zakoni."
        ],
        "keywords": ["registracija", "udruženje", "fondacija", "registar"]
    },
    "obrasci": {
        "responses": [
            "Obrasci za registraciju udruženja dostupni su na https://www.mpr.gov.ba/bs/obrasci-za-registraciju.",
            "Potreban vam je obrazac za preregistraciju? Pogledajte https://www.mpr.gov.ba/bs/preregistracija.",
            "Svi obrasci su dostupni u PDF formatu. Možete ih preuzeti ili popuniti online."
        ],
        "keywords": ["obrazac", "formular", "obrazci", "preuzmi"]
    },
    "zakoni": {
        "responses": [
            "Zakon o udrugama reguliše osnivanje i rad nevladinih organizacija. Dostupan na https://www.mpr.gov.ba/bs/zakon-o-udrugama.",
            "Zakon o zaštiti ličnih podataka implementira GDPR. Detaljno: https://www.mpr.gov.ba/bs/zakon-o-zastiti.",
            "Sve zakone i pravilnike pronađite u sekciji Zakoni i pravilnici."
        ],
        "keywords": ["zakon", "pravilnik", "pravna", "regula"]
    },
    "procedure": {
        "responses": [
            "Procedura za registraciju: 1) Priprema dokumenata, 2) Podnošenje zahtjeva, 3) Provjera, 4) Odobrenje.",
            "Procedura je dostupna na https://www.mpr.gov.ba/bs/procedure-i-rokovi. Prosječno traje 20 dana.",
            "Za brže rješavanje kontaktirajte službu korisnika na +387 33 XXX XXXX."
        ],
        "keywords": ["procedura", "koraci", "etape", "rok", "vrijeme"]
    }
}

DEFAULT_RESPONSE = (
    "Hvala na pitanju! Informacija koju tražite možda je dostupna u našoj bazi znanja. "
    "Pretraživajte po ključnim pojmovima ili posjetite https://www.mpr.gov.ba za više detalja. "
    "Ako trebate brzu pomoć, kontaktirajte ministarstvo direktno."
)


def get_dummy_response(query: str) -> Dict[str, any]:
    """
    Generate a dummy response based on query keywords.
    Returns dict with response text, confidence, and source info.
    """
    query_lower = query.lower()
    
    # Try to match keywords
    matched_category = None
    highest_match = 0
    
    for category, data in DUMMY_KB.items():
        match_count = sum(1 for kw in data["keywords"] if kw in query_lower)
        if match_count > highest_match:
            highest_match = match_count
            matched_category = category
    
    # Return response
    if matched_category:
        response_text = random.choice(DUMMY_KB[matched_category]["responses"])
        confidence = min(0.9, 0.5 + (highest_match * 0.2))
    else:
        response_text = DEFAULT_RESPONSE
        confidence = 0.4
    
    return {
        "response": response_text,
        "confidence": confidence,
        "category": matched_category or "general",
        "source": "dummy_kb",
        "timestamp": None
    }


def get_suggested_questions() -> List[str]:
    """Return suggested questions for the UI."""
    return [
        "Kako registrovati udruženje?",
        "Koji su potrebni obrasci?",
        "Gdje mogu pronaći zakone?",
        "Koja je procedura za registraciju?",
        "Kako kontaktirati ministarstvo?"
    ]
