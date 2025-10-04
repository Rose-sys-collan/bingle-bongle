import os
import json
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL = "models/gemini-2.5-flash"

MISTAKE_SCHEMA = {
  "type": "object",
  "properties": {
    "mistakes": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "span": {"type": "string"},
          "type": {"type": "string"},
          "fix": {"type": "string"},
          "severity": {"type": "integer"}   # ← 去掉 minimum / maximum
        },
        "required": ["span", "fix", "severity"]
      }
    }
  },
  "required": ["mistakes"]
}

def analyze_mistakes(note_text: str):
  model = genai.GenerativeModel(MODEL)
  resp = model.generate_content(
    note_text,
    generation_config={
      "response_mime_type": "application/json",
      "response_schema": MISTAKE_SCHEMA
    }
  )
  return json.loads(resp.text)


# ---------- Feature ②: Translate + Plain English ----------
import json as _json  # 防止上面没有导入json时出错；若已有import json也没关系

TRANSLATE_SCHEMA = {
  "type": "object",
  "properties": {
    "translation": {"type": "string"},
    "plain_en": {"type": "string"},
    "examples": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["translation", "plain_en"]
}

def translate_with_explain(text: str, target_lang: str):
    """
    Translate text into target_lang (e.g., 'zh-CN' or 'en'),
    and explain key terms in Plain English with 1–2 examples.
    Returns a dict per TRANSLATE_SCHEMA.
    """
    model = genai.GenerativeModel(MODEL)
    prompt = (
      "You are a bilingual teaching assistant. "
      "Translate between English and Simplified Chinese as requested. "
      "Also explain key terms in CEFR-B1 Plain English and include 1–2 concrete usage examples.\n\n"
      f"Target language: {target_lang}\n"
      f"Text:\n{text}\n\n"
      "Return ONLY valid JSON per schema."
    )
    resp = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": TRANSLATE_SCHEMA
        }
    )
    return _json.loads(resp.text)

def translate_with_explain(text: str, target_lang: str):
    """
    Translate text into target_lang (e.g., 'zh-CN' or 'en'),
    and explain key terms in Plain English with 1–2 examples.
    Returns a dict per TRANSLATE_SCHEMA.
    """
    model = genai.GenerativeModel(MODEL)
    prompt = (
      "You are a bilingual teaching assistant. "
      "Translate between English and Simplified Chinese as requested. "
      "Also explain key terms in CEFR-B1 Plain English and include 1–2 concrete usage examples.\n"
      "Use CEFR-B1 vocabulary; keep plain_en ≤ 2 sentences.\n"
      "Return at most 2 examples, each ≤ 1 sentence.\n\n"
      f"Target language: {target_lang}\n"
      f"Text:\n{text}\n\n"
      "Return ONLY valid JSON per schema."
    )
    resp = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": TRANSLATE_SCHEMA
        }
    )
    return json.loads(resp.text)

# ---------- Feature ④: Make study cards (Anki/Quizlet) ----------
import json as _json  # 若上面已 import json 也不会冲突

CARDS_SCHEMA = {
  "type": "object",
  "properties": {
    "cards": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "term": {"type": "string"},
          "definition": {"type": "string"}
        },
        "required": ["term", "definition"]
      }
    }
  },
  "required": ["cards"]
}

def make_cards(note_text: str, limit: int=10):
    model = genai.GenerativeModel(MODEL)
    prompt = (
      f"Create up to {limit} concise term-definition pairs from the note below.\n"
      "Each definition ≤ 2 sentences, concrete, self-contained.\n"
      "Return ONLY valid JSON per schema.\n\n"
      f"{note_text}"
    )
    resp = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": CARDS_SCHEMA
        }
    )
    return _json.loads(resp.text)

def to_anki_tsv(cards_json: dict) -> str:
    return "\n".join(
        f"{c.get('term','')}\t{c.get('definition','')}"
        for c in cards_json.get("cards", [])
    )

# ---------- Feature ③: Convert handwritten math to LaTeX ----------
import json as _json  # 若上面已 import json 也无冲突

LATEX_SCHEMA = {
  "type": "object",
  "properties": {
    "latex": {"type": "string"},
    "description": {"type": "string"}
  },
  "required": ["latex"]
}

def image_to_latex(math_text: str):
    """
    Convert a handwritten or plain math expression into LaTeX code.
    If the input is already typed, just standardize it.
    Returns: dict with 'latex' and optional 'description'.
    """
    model = genai.GenerativeModel(MODEL)
    prompt = (
      "You are a math notation assistant. "
      "Given a handwritten or plain text math expression, convert it into valid LaTeX syntax. "
      "Keep it minimal (no extra spaces), and escape LaTeX properly. "
      "Also include a short plain English description of what the expression means.\n\n"
      f"Expression:\n{math_text}\n\n"
      "Return ONLY JSON per schema."
    )
    resp = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": LATEX_SCHEMA
        }
    )
    return _json.loads(resp.text)


PROMPT_NOTE_REWRITE = """
You are EchoClass, an AI note rewriter that helps ESL (English-as-a-Second-Language)
students express their ideas clearly and inclusively.

Task:
- Rewrite the following classroom note into fluent, simple, and unbiased English.
- Keep the academic meaning, but make the tone respectful and inclusive.
- Replace biased words or phrases (e.g., “foreign students”, “poor English”) with neutral alternatives.
- Output only the rewritten text, without explanation.

Note:
{note}
"""

def rewrite_note(text):
    prompt = PROMPT_NOTE_REWRITE.format(note=text)
    # 调用模型接口，Gemini
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text


test_notes = [
    "Foreign students often struggle to present clearly.",
    "Some people speak bad English but try their best.",
    "The professor said girls are better at languages."
]

for n in test_notes:
    result = rewrite_note(n)
    print("📝", n)
    print("✨", result)
    print("-"*60)

# ========== Demo helper functions  ==========

def rewrite_note(text: str) -> str:
    """
    Rewrites a classroom note to be simpler and bias-free.
    (This is a placeholder; replace with your real model call later.)
    """
    rules = [
        ("foreign students", "international students"),
        ("poor English", "English in progress"),
        ("girls are better at", "students are often encouraged to develop strength in"),
        ("hard", "challenging"),
        ("too fast", "a bit fast"),
    ]
    out = text
    for a, b in rules:
        out = out.replace(a, b)
    if out == text:
        out = "This note has been rewritten in clear and inclusive English, removing biased wording."
    return out


def run_demo(text: str):
    """Quick demo for Jupyter notebook display."""
    original = text.strip()
    rewritten = rewrite_note(original)
    print("📝 Original:", original)
    print("✨ Rewritten:", rewritten)
    return {"original": original, "rewritten": rewritten}
