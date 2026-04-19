import os
from pathlib import Path


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


def print_provider_status() -> None:
    load_dotenv()

    provider_mode = os.getenv("AI_PROVIDER", "auto").lower()
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "")
    hf_model = os.getenv("HF_MODEL", "")

    print(f"AI_PROVIDER={provider_mode}")
    print(f"GEMINI configured={bool(os.getenv('GEMINI_API_KEY'))} model={gemini_model}")
    print(
        f"OPENROUTER configured={bool(os.getenv('OPENROUTER_API_KEY'))} "
        f"model={openrouter_model or '(not set)'}"
    )
    print(f"HUGGINGFACE configured={bool(os.getenv('HF_API_KEY'))} model={hf_model or '(not set)'}")
    print("LOCAL configured=True model=rule-based")


if __name__ == "__main__":
    print_provider_status()
