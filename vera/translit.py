"""Hinglish (Roman) → Hindi (Devanagari) for Vera's own phrasing.

Vera's Hindi wording is a closed vocabulary, so a word map is exact and deterministic (no LLM needed). Only Hindi
words are converted; names, prices, numbers, dates, English service terms and CTA keywords (YES / GO / STOP) stay in
Latin script — the way Hindi WhatsApp messages are actually written in India.
"""
from __future__ import annotations

import re

HI = {
    # function words
    "ke": "के", "ka": "का", "ki": "की", "ko": "को", "se": "से", "mein": "में", "pe": "पे", "par": "पर", "aur": "और",
    "ya": "या", "bhi": "भी", "toh": "तो", "tak": "तक", "liye": "लिए", "saath": "साथ", "baad": "बाद",
    "pehle": "पहले", "abhi": "अभी", "aaj": "आज", "kal": "कल", "ab": "अब", "jab": "जब", "tab": "तब", "phir": "फिर",
    "sirf": "सिर्फ़", "bas": "बस", "har": "हर", "sab": "सब", "sabhi": "सभी", "kuch": "कुछ", "koi": "कोई", "ek": "एक",
    "na": "न", "nahi": "नहीं", "jo": "जो", "woh": "वो", "yeh": "यह", "iska": "इसका", "iski": "इसकी", "ise": "इसे",
    "isi": "इसी", "isliye": "इसलिए", "un": "उन", "unhe": "उन्हें", "unki": "उनकी", "unka": "उनका", "unke": "उनके",
    "unmein": "उनमें", "kis": "किस", "kya": "क्या", "kitne": "कितने", "jitne": "जितने", "kai": "कई", "jaise": "जैसे",
    "jaisa": "जैसा", "aisa": "ऐसा", "aise": "ऐसे", "aisi": "ऐसी", "wahi": "वही", "yahin": "यहीं", "yahan": "यहाँ",
    "upar": "ऊपर", "neeche": "नीचे", "aage": "आगे", "wapas": "वापस", "lekin": "लेकिन", "shayad": "शायद", "taaki": "ताकि",
    "bina": "बिना", "baare": "बारे", "paas": "पास", "aas-paas": "आस-पास", "lagbhag": "लगभग", "zyada": "ज़्यादा",
    "kam": "कम", "thoda": "थोड़ा", "sahi": "सही", "main": "मैं", "mera": "मेरा", "mere": "मेरे", "hum": "हम",
    "aap": "आप", "aapka": "आपका", "aapke": "आपके", "aapki": "आपकी", "aapse": "आपसे", "aapne": "आपने", "apna": "अपना",
    "apne": "अपने", "apni": "अपनी", "bajaye": "बजाय", "saamne": "सामने", "ji": "जी", "kabhi": "कभी", "sach": "सच",
    "pakka": "पक्का", "zaroor": "ज़रूर", "zaroori": "ज़रूरी", "seedhi": "सीधी", "ek-ek": "एक-एक", "baar-baar": "बार-बार",
    "baar": "बार", "wali": "वाली", "wala": "वाला", "wale": "वाले", "matlab": "मतलब", "iska": "इसका", "kaha": "कहा",
    # verbs / auxiliaries
    "hai": "है", "hain": "हैं", "hoon": "हूँ", "ho": "हो", "tha": "था", "thi": "थी", "hui": "हुई", "hue": "हुए",
    "hota": "होता", "hote": "होते", "hona": "होना", "hoga": "होगा", "gaya": "गया", "gaye": "गए", "kar": "कर",
    "karna": "करना", "karne": "करने", "karein": "करें", "karte": "करते", "karti": "करती", "karta": "करता",
    "karega": "करेगा", "karegi": "करेगी", "karenge": "करेंगे", "karungi": "करूँगी", "karke": "करके", "kiya": "किया",
    "kiye": "किए", "kijiye": "कीजिए", "dijiye": "दीजिए", "doon": "दूँ", "do": "दो", "di": "दी", "deti": "देती",
    "degi": "देगी", "denge": "देंगे", "dungi": "दूँगी", "lenge": "लेंगे", "liya": "लिया", "lete": "लेते",
    "loongi": "लूँगी", "bhej": "भेज", "bhejiye": "भेजिए", "bhejein": "भेजें", "bhejo": "भेजो", "bana": "बना",
    "banata": "बनाता", "banenge": "बनेंगे", "dekh": "देख", "dekhe": "देखे", "dekhte": "देखते", "dekhne": "देखने",
    "dikhaye": "दिखाए", "dikhta": "दिखता", "dikhne": "दिखने", "lagta": "लगता", "laga": "लगा", "lag": "लग",
    "chahiye": "चाहिए", "chahein": "चाहें", "sakti": "सकती", "sakte": "सकते", "sakta": "सकता", "sakein": "सकें",
    "paungi": "पाऊँगी", "rahi": "रही", "rahe": "रहे", "raha": "रहा", "rehte": "रहते", "rehna": "रहना",
    "rahenge": "रहेंगे", "rahein": "रहें", "chalte": "चलते", "chal": "चल", "ja": "जा", "jaata": "जाता",
    "jaati": "जाती", "jaaye": "जाए", "jaayega": "जाएगा", "jaayegi": "जाएगी", "jaayenge": "जाएँगे", "aa": "आ",
    "aaya": "आया", "aayi": "आई", "aate": "आते", "aayega": "आएगा", "aayein": "आएँ", "mile": "मिले", "milte": "मिलते",
    "milti": "मिलती", "bataiye": "बताइए", "batayein": "बताएँ", "bata": "बता", "bataungi": "बताऊँगी",
    "poochiye": "पूछिए", "pooch": "पूछ", "poocha": "पूछा", "rakhein": "रखें", "sunte": "सुनते", "badal": "बदल",
    "badalta": "बदलता", "badhane": "बढ़ाने", "badhe": "बढ़े", "badhayein": "बढ़ाएँ", "girenge": "गिरेंगे",
    "girne": "गिरने", "utar": "उतर", "nikaal": "निकाल", "daal": "डाल", "jud": "जुड़", "dhoondh": "ढूँढ",
    "dhoondhenge": "ढूँढेंगे", "kheenchte": "खींचते", "pahunchne": "पहुँचने", "sambhalne": "सँभालने",
    "sambhalti": "सँभालती", "sambhal": "सँभाल", "rokna": "रोकना", "bachaana": "बचाना", "bachaani": "बचानी",
    "bache": "बचे", "khula": "खुला", "shuru": "शुरू", "hona": "होना",
    # nouns / adjectives
    "din": "दिन", "hafte": "हफ़्ते", "hafton": "हफ़्तों", "mahine": "महीने", "log": "लोग", "baat": "बात",
    "kaam": "काम", "cheez": "चीज़", "cheezon": "चीज़ों", "naam": "नाम", "madad": "मदद", "jaankari": "जानकारी",
    "khushi": "ख़ुशी", "pareshani": "परेशानी", "jawab": "जवाब", "sawaal": "सवाल", "accha": "अच्छा",
    "badhiya": "बढ़िया", "jaldi": "जल्दी", "doosre": "दूसरे", "dusron": "दूसरों", "naya": "नया", "naye": "नए",
    "pehla": "पहला", "pehli": "पहली", "pichhle": "पिछले", "pichhli": "पिछली", "agla": "अगला", "baaki": "बाकी",
    "sabse": "सबसे", "chhota": "छोटा", "aasaan": "आसान", "sasta": "सस्ता", "tareeka": "तरीका", "tareef": "तारीफ़",
    "pasand": "पसंद", "fark": "फ़र्क", "wajah": "वजह", "waada": "वादा", "mehnat": "मेहनत", "shaadi": "शादी",
    "dawai": "दवाई", "mulaqat": "मुलाक़ात", "intezaar": "इंतज़ार", "girawat": "गिरावट", "badhat": "बढ़त",
    "mukable": "मुकाबले", "jagah": "जगह", "layak": "लायक", "dheema": "धीमा", "khatam": "ख़त्म", "sahi": "सही",
    "kahun": "कहूँ", "door": "दूर", "diya": "दिया", "walon": "वालों", "pahunchti": "पहुँचती", "badlaav": "बदलाव",
    "bhejti": "भेजती", "hafta": "हफ़्ता", "aapko": "आपको", "mujhe": "मुझे", "humein": "हमें", "isko": "इसको",
    "unko": "उनको", "kaise": "कैसे", "kaisa": "कैसा", "kyunki": "क्योंकि", "agar": "अगर", "dono": "दोनों",
    "poora": "पूरा", "pura": "पूरा", "bahut": "बहुत", "theek": "ठीक", "haan": "हाँ", "achha": "अच्छा", "yahi": "यही",
    "milega": "मिलेगा", "milenge": "मिलेंगे", "chalenge": "चलेंगे", "aayenge": "आएँगे", "aayengi": "आएँगी",
    "sakta": "सकता", "lagega": "लगेगा", "rakhna": "रखना", "rakh": "रख", "laga": "लगा", "daalein": "डालें", "jinhone": "जिन्होंने", "kare": "करे", "unhone": "उन्होंने", "nahin": "नहीं", "ghante": "घंटे",
    "bilkul": "बिल्कुल", "samajh": "समझ", "gayi": "गई", "koshish": "कोशिश", "hamesha": "हमेशा", "dobara": "दोबारा",
    "rok": "रोक", "rakhti": "रखती", "badalni": "बदलनी", "chalega": "चलेगा", "kijiye": "कीजिए", "karunga": "करूँगा",
    "hoon": "हूँ", "dhanyavaad": "धन्यवाद", "shukriya": "शुक्रिया", "mat": "मत", "band": "बंद", "khushi": "ख़ुशी", "hua": "हुआ", "hui": "हुई",
}

_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


def to_devanagari(text: str, protect: tuple = ()) -> str:
    """Convert Vera's Hinglish phrasing to Devanagari. `protect` = names/offers/places from the contexts, kept verbatim;
    anything not in the map (English terms, numbers, YES/GO) is kept as-is too."""
    text = text or ""
    keep: list[str] = []
    for ph in sorted({p for p in protect if p and len(p) > 1}, key=len, reverse=True):
        if ph in text:
            keep.append(ph)
            text = text.replace(ph, f"\x00{len(keep) - 1}\x00")

    # "is" = इस only before a Hindi noun; "hi" = ही only mid-sentence after a word (never the greeting "Hi")
    text = re.sub(r"\b[Ii]s\s+(?=(hafte|saal|mahine|window|baar|din|season|post|offer|trend|week|baare|tarah)\b)", "इस ", text)
    text = re.sub(r"(?<=[a-z\u0900-\u097F]) hi\b(?!\s*[!,])", " ही", text)

    def sub(m: re.Match) -> str:
        w = m.group(0)
        return HI.get(w.lower()) or w
    out = _WORD.sub(sub, text)
    return re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], out)


# incoming Devanagari → Roman, so the intent rules (written for Roman Hinglish) understand Hindi-script replies
ROMAN = {v: k for k, v in HI.items()}
ROMAN.update({"हाँ": "haan", "हां": "haan", "हा": "haan", "जी": "ji", "ठीक": "theek", "बिल्कुल": "bilkul", "चलो": "chalo",
              "करो": "karo", "कर": "kar", "दो": "do", "दीजिए": "dijiye", "दीजिये": "dijiye", "भेजो": "bhejo", "भेज": "bhej",
              "नहीं": "nahi", "नही": "nahi", "मत": "mat", "बंद": "band", "रुको": "ruko", "रोको": "roko", "बाद": "baad",
              "अभी": "abhi", "व्यस्त": "busy", "कौन": "kaun", "कितना": "kitna", "कितने": "kitne", "पैसे": "paise",
              "धन्यवाद": "dhanyavaad", "शुक्रिया": "shukriya", "रुचि": "ruchi", "चाहिए": "chahiye", "हो": "ho", "गया": "gaya",
              "मुझे": "mujhe", "जुड़ना": "judna", "जुड़ना": "judna", "जोड़ो": "jodo", "आइडिया": "ideas", "ड्राफ्ट": "draft", "कहाँ": "kahan"})
_DEV = re.compile(r"[\u0900-\u097F]+")


def to_roman(text: str) -> str:
    return _DEV.sub(lambda m: ROMAN.get(m.group(0), m.group(0)), text or "")
