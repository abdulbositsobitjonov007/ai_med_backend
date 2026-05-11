import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import google.genai as genai
from pydantic import BaseModel, Field


def load_dotenv() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()

DEFAULT_PROVIDER = os.getenv("AI_PROVIDER", "auto").lower()
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "")
DEFAULT_HF_MODEL = os.getenv("HF_MODEL", "")

LANGUAGE_NAMES = {
    "ru": "Russian",
    "en": "English",
    "uz": "Uzbek",
}

LOCAL_MESSAGES = {
    "ru": {
        "danger_reason": "Есть признаки потенциально опасного состояния.",
        "danger_advice": "Срочно обратитесь за неотложной медицинской помощью.",
        "infection_reason": "Симптомы инфекции на фоне температуры требуют наблюдения и могут потребовать консультации врача.",
        "infection_advice": "Пейте больше жидкости, отдыхайте, контролируйте температуру и обратитесь к врачу, если станет хуже.",
        "mild_reason": "По описанию нет явных признаков срочности, но оценка ограничена.",
        "mild_advice": "Наблюдайте за состоянием и обратитесь к врачу при ухудшении.",
        "override_reason": "Боль в горле или другие симптомы инфекции на фоне температуры требуют наблюдения и обычно не относятся к полностью безопасной категории.",
        "override_advice": "Пейте больше жидкости, отдыхайте, контролируйте температуру и обратитесь к врачу, если станет хуже или температура вырастет.",
        "error_reason": "Ошибка связи с ИИ",
    },
    "en": {
        "danger_reason": "There are signs of a potentially dangerous condition.",
        "danger_advice": "Seek urgent medical care immediately.",
        "infection_reason": "Infection symptoms together with fever require monitoring and may need medical advice.",
        "infection_advice": "Drink more fluids, rest, monitor your temperature, and contact a doctor if you get worse.",
        "mild_reason": "There are no obvious urgent warning signs from the description, but this assessment is limited.",
        "mild_advice": "Monitor your condition and seek medical advice if symptoms worsen.",
        "override_reason": "Sore throat or other infection symptoms together with fever usually should not be treated as fully low-risk.",
        "override_advice": "Drink more fluids, rest, monitor your temperature, and contact a doctor if you worsen or the temperature rises.",
        "error_reason": "AI connection error",
    },
    "uz": {
        "danger_reason": "Xavfli holat belgilari bo'lishi mumkin.",
        "danger_advice": "Zudlik bilan shoshilinch tibbiy yordamga murojaat qiling.",
        "infection_reason": "Harorat bilan kechayotgan infeksiya alomatlari kuzatuvni va shifokor maslahatini talab qilishi mumkin.",
        "infection_advice": "Ko'proq suyuqlik iching, dam oling, haroratni kuzating va ahvol yomonlashsa shifokorga murojaat qiling.",
        "mild_reason": "Ta'rif bo'yicha shoshilinch xavf belgilari ko'rinmayapti, lekin bu baholash cheklangan.",
        "mild_advice": "Holatingizni kuzating va alomatlar kuchaysa shifokorga murojaat qiling.",
        "override_reason": "Tomoq og'rig'i yoki boshqa infeksiya alomatlari harorat bilan birga bo'lsa, bu odatda to'liq xavfsiz holat hisoblanmaydi.",
        "override_advice": "Ko'proq suyuqlik iching, dam oling, haroratni kuzating va ahvol yomonlashsa yoki harorat ko'tarilsa shifokorga murojaat qiling.",
        "error_reason": "Sun'iy intellekt bilan aloqa xatosi",
    },
}

SYSTEM_PROMPT_TEMPLATE = """
You are a careful medical triage assistant.
Analyze the patient's symptoms and respond with STRICTLY valid JSON.
Use conservative triage:
- RED for severe danger signs such as chest pain, severe shortness of breath, heavy bleeding, confusion, fainting, stroke-like symptoms, or suspected emergency.
- YELLOW for symptoms that usually need medical advice or observation within 24 hours, including sore throat with fever, cough with fever, weakness with fever, persistent pain, worsening infection symptoms, or anything that should not be dismissed as mild.
- GREEN only for clearly mild self-limited symptoms without danger signs and without concerning combinations.
If the patient mentions sore throat plus temperature/fever, do not return GREEN.
Respond in the same language as the patient's message. The detected patient language is: {language_name}.
Schema:
{{
  "color": "RED|YELLOW|GREEN",
  "reason": "short explanation in the same language as the user",
  "advice": "what to do next in the same language as the user"
}}
Do not include markdown fences or extra text.
""".strip()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientQuery(BaseModel):
    symptoms: str = Field(..., min_length=3)


def detect_language(text: str) -> str:
    lowered = text.lower()

    uzbek_markers = [
        " men ",
        " menda ",
        " tomoq",
        " isitma",
        " harorat",
        " yo'tal",
        " og'ri",
        " ogri",
        " ko'ngil",
        " kongil",
        " bezovta",
        " bosh ",
        " qorin",
        " nafas",
        "g'",
        "o'",
        "shifokor",
    ]
    english_markers = [
        " i ",
        " my ",
        " have ",
        " sore throat",
        " fever",
        " cough",
        " pain",
        " headache",
        " weak",
        " tired",
        " breathing",
        " chest",
    ]

    if any("а" <= char <= "я" or char == "ё" for char in lowered):
        return "ru"
    if any(marker in f" {lowered} " for marker in uzbek_markers):
        return "uz"
    if any(marker in f" {lowered} " for marker in english_markers):
        return "en"
    return "uz" if any(char in lowered for char in ["o'", "g'", "q", "x"]) else "en"


def messages_for(text: str) -> dict[str, str]:
    return LOCAL_MESSAGES[detect_language(text)]


def build_system_prompt(symptoms: str) -> str:
    language_code = detect_language(symptoms)
    language_name = LANGUAGE_NAMES[language_code]
    return SYSTEM_PROMPT_TEMPLATE.format(language_name=language_name)


def parse_json_response(raw_text: str) -> dict[str, str]:
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(cleaned)

    if not isinstance(parsed, dict):
        raise ValueError("Model response is not a JSON object")

    color = str(parsed.get("color", "GRAY")).upper()
    if color not in {"RED", "YELLOW", "GREEN"}:
        color = "GRAY"

    return {
        "color": color,
        "reason": str(parsed.get("reason", "No reason provided")),
        "advice": str(parsed.get("advice", "No advice provided")),
    }


def build_user_prompt(symptoms: str) -> str:
    return f'Patient symptoms: "{symptoms}"'


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def enforce_minimum_triage(symptoms: str, result: dict[str, str]) -> dict[str, str]:
    text = symptoms.lower()
    strings = messages_for(symptoms)

    red_keywords = [
        "сердце",
        "боль в груди",
        "не дыш",
        "задыха",
        "кровь",
        "потерял сознание",
        "обморок",
        "судорог",
        "паралич",
        "инсульт",
        "chest pain",
        "can't breathe",
        "cannot breathe",
        "faint",
        "seizure",
        "stroke",
        "nafas",
        "hushdan",
        "ko'krak",
        "kokrak",
    ]
    fever_keywords = ["температур", "лихорад", "жар", "37.", "38", "39", "40", "fever", "temperature", "isitma", "harorat"]
    infection_keywords = ["горло", "кашель", "слабость", "озноб", "глотать больно", "sore throat", "cough", "weakness", "tomoq", "yo'tal", "yotal", "holsiz"]

    if contains_any(text, red_keywords):
        result["color"] = "RED"
        if result.get("reason", "").strip() in {"", "No reason provided"}:
            result["reason"] = strings["danger_reason"]
        if result.get("advice", "").strip() in {"", "No advice provided"}:
            result["advice"] = strings["danger_advice"]
        return result

    if contains_any(text, fever_keywords) and contains_any(text, infection_keywords) and result.get("color") == "GREEN":
        result["color"] = "YELLOW"
        result["reason"] = strings["override_reason"]
        result["advice"] = strings["override_advice"]

    return result


def gemini_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def openrouter_available() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY") and DEFAULT_OPENROUTER_MODEL)


def huggingface_available() -> bool:
    return bool(os.getenv("HF_API_KEY") and DEFAULT_HF_MODEL)


def analyze_with_gemini(symptoms: str) -> dict[str, str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=DEFAULT_GEMINI_MODEL,
        contents=f"{build_system_prompt(symptoms)}\n\n{build_user_prompt(symptoms)}",
    )
    result = parse_json_response(response.text or "")
    result = enforce_minimum_triage(symptoms, result)
    result["provider"] = "gemini"
    result["model"] = DEFAULT_GEMINI_MODEL
    return result


def post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc

    return json.loads(body)


def analyze_with_openrouter(symptoms: str) -> dict[str, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    if not DEFAULT_OPENROUTER_MODEL:
        raise RuntimeError("OPENROUTER_MODEL is not set")

    payload = {
        "model": DEFAULT_OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": build_system_prompt(symptoms)},
            {"role": "user", "content": build_user_prompt(symptoms)},
        ],
    }
    response = post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        payload,
        {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "ai_project_medical_triage",
        },
    )
    content = response["choices"][0]["message"]["content"]
    result = parse_json_response(content)
    result = enforce_minimum_triage(symptoms, result)
    result["provider"] = "openrouter"
    result["model"] = DEFAULT_OPENROUTER_MODEL
    return result


def analyze_with_huggingface(symptoms: str) -> dict[str, str]:
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise RuntimeError("HF_API_KEY is not set")
    if not DEFAULT_HF_MODEL:
        raise RuntimeError("HF_MODEL is not set")

    prompt = f"{build_system_prompt(symptoms)}\n\n{build_user_prompt(symptoms)}\nJSON:"
    response = post_json(
        f"https://api-inference.huggingface.co/models/{DEFAULT_HF_MODEL}",
        {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 180,
                "return_full_text": False,
            },
        },
        {"Authorization": f"Bearer {api_key}"},
    )

    if isinstance(response, list) and response and "generated_text" in response[0]:
        raw_text = response[0]["generated_text"]
    else:
        raise RuntimeError(f"Unexpected Hugging Face response: {response}")

    result = parse_json_response(raw_text)
    result = enforce_minimum_triage(symptoms, result)
    result["provider"] = "huggingface"
    result["model"] = DEFAULT_HF_MODEL
    return result


def analyze_locally(symptoms: str) -> dict[str, str]:
    text = symptoms.lower()
    strings = messages_for(symptoms)

    if contains_any(
        text,
        [
            "сердце",
            "боль в груди",
            "не дыш",
            "задыха",
            "кровь",
            "потерял сознание",
            "обморок",
            "chest pain",
            "can't breathe",
            "cannot breathe",
            "bleeding",
            "fainted",
            "nafas",
            "hushdan",
            "ko'krak",
            "kokrak",
        ],
    ):
        return {
            "color": "RED",
            "reason": strings["danger_reason"],
            "advice": strings["danger_advice"],
            "provider": "local",
            "model": "rule-based",
        }

    if contains_any(text, ["температур", "лихорад", "жар", "37.", "38", "39", "fever", "temperature", "isitma", "harorat"]) and contains_any(
        text,
        ["горло", "кашель", "слабость", "озноб", "насморк", "sore throat", "cough", "weakness", "runny nose", "tomoq", "yo'tal", "yotal", "holsiz"],
    ):
        return {
            "color": "YELLOW",
            "reason": strings["infection_reason"],
            "advice": strings["infection_advice"],
            "provider": "local",
            "model": "rule-based",
        }

    return {
        "color": "GREEN",
        "reason": strings["mild_reason"],
        "advice": strings["mild_advice"],
        "provider": "local",
        "model": "rule-based",
    }


def get_provider_order() -> list[str]:
    if DEFAULT_PROVIDER != "auto":
        return [DEFAULT_PROVIDER]

    order: list[str] = []
    if gemini_available():
        order.append("gemini")
    if openrouter_available():
        order.append("openrouter")
    if huggingface_available():
        order.append("huggingface")
    order.append("local")
    return order


def analyze_with_provider(provider: str, symptoms: str) -> dict[str, str]:
    if provider == "gemini":
        return analyze_with_gemini(symptoms)
    if provider == "openrouter":
        return analyze_with_openrouter(symptoms)
    if provider == "huggingface":
        return analyze_with_huggingface(symptoms)
    if provider == "local":
        return analyze_locally(symptoms)
    raise RuntimeError(f"Unsupported provider: {provider}")


@app.post("/analyze")
async def analyze_symptoms(data: PatientQuery) -> dict:
    errors: list[str] = []
    strings = messages_for(data.symptoms)
    provider_order = get_provider_order()

    for i, provider in enumerate(provider_order):
        try:
            result = analyze_with_provider(provider, data.symptoms)
            # If we fell back past the first choice, include a warning
            if i > 0 and errors:
                result["warnings"] = " | ".join(errors)
            return result
        except Exception as exc:
            errors.append(f"{provider}: {exc}")

    return {
        "color": "GRAY",
        "reason": strings["error_reason"],
        "advice": " ; ".join(errors) if errors else "No providers available",
        "provider": "none",
        "model": "none",
        "warnings": " | ".join(errors),
    }


@app.get("/")
@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "Medical AI Backend is online",
        "provider_mode": DEFAULT_PROVIDER,
        "provider_order": get_provider_order(),
        "configured": {
            "gemini": gemini_available(),
            "openrouter": openrouter_available(),
            "huggingface": huggingface_available(),
            "local": True,
        },
        "models": {
            "gemini": DEFAULT_GEMINI_MODEL,
            "openrouter": DEFAULT_OPENROUTER_MODEL or "(not set)",
            "huggingface": DEFAULT_HF_MODEL or "(not set)",
        },
    }


@app.get("/providers")
def check_providers() -> dict[str, object]:
    """Diagnostic endpoint: tests each configured provider and reports status."""
    results: dict[str, object] = {}
    test_symptoms = "I have a headache"

    for provider in get_provider_order():
        try:
            res = analyze_with_provider(provider, test_symptoms)
            results[provider] = {"status": "ok", "model": res.get("model", "?")}
        except Exception as exc:
            results[provider] = {"status": "error", "detail": str(exc)[:300]}

    return {"providers": results, "order": get_provider_order()}
