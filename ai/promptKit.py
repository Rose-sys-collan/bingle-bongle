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


# ==== EchoClass: Model-backed rewrite (Gemini) ====
import os, re
from typing import Optional

# 读 .env（若不存在也不会报错）
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# 尝试导入 Gemini SDK
_GENAI_OK = True
try:
    import google.generativeai as genai
except Exception:
    _GENAI_OK = False

# ---- Prompt 模板（可改写）----
PROMPT_NOTE_REWRITE = """
You are EchoClass, an unbiased classroom assistant that rewrites notes for ESL students.

Rewrite the note below into clear, concise, and inclusive English:
- Preserve academic meaning and intent; do not invent facts.
- Avoid stereotypes or evaluative wording (e.g., “foreign”, “poor English”).
- Prefer short sentences and common words.
- If the note contains multiple items, keep a simple numbered list.
- Output only the rewritten text, with no preface or explanations.

Note:
{note}
"""

# ---- 规则版后备（无 Key/超限时不崩）----
def _rule_based_fallback(text: str) -> str:
    s = (text or "").strip()
    repl = [
        (r"\bforeign students\b", "international students"),
        (r"\bstudents with (strong|thick) accents\b", "students who speak with diverse accents"),
        (r"\bpoor English\b", "English in progress"),
        (r"\bbad English\b", "English in progress"),
        (r"\btoo fast\b", "a bit fast"),
        (r"\bhard words\b", "complex words"),
        (r"\bstruggle to present\b", "may face challenges presenting"),
    ]
    for a, b in repl:
        s = re.sub(a, b, s, flags=re.IGNORECASE)
    if re.search(r"\baccents\b", s, re.I) and re.search(r"\bpresent", s, re.I):
        if not re.search(r"can .*present .*effectively", s, re.I):
            s = s.rstrip(".") + ", and with inclusive feedback and practice, they can present effectively."
    s = re.sub(r"\s+", " ", s).strip()
    return s or "This note has been rewritten in clear and inclusive English."

# ---- 核心：调用模型的改写函数 ----
def rewrite_note(
    text: str,
    template: Optional[str] = None,
    use_model: bool = True,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """Rewrite note with Gemini (fallback to rule-based when unavailable)."""
    note = (text or "").strip()
    if not note:
        return ""

    # 默认从环境变量读取模型与温度
    model_name = model_name or os.getenv("ECHOCLASS_MODEL", "gemini-1.5-flash")
    try:
        temperature = float(temperature if temperature is not None else os.getenv("ECHOCLASS_TEMPERATURE", "0.4"))
    except Exception:
        temperature = 0.4

    # 如果没要求用模型，或者 SDK/Key 不可用 → 直接走后备
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if (not use_model) or (not _GENAI_OK) or (not api_key):
        return _rule_based_fallback(note)

    # 配置并调用
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = (template or PROMPT_NOTE_REWRITE).format(note=note)
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
            },
        )
        out = getattr(resp, "text", "") or ""
        out = re.sub(r"\s+", " ", out).strip()
        # 有些模型会加解释性前缀，这里保险剥离一下
        out = re.sub(r"^(Rewrite[d]?|Rewriting|Here is|Here’s|Output)\s*[:\-–]\s*", "", out, flags=re.I)
        return out or _rule_based_fallback(note)
    except Exception:
        # 任何网络/配额/安全阻断问题 → 回退规则版，不让体验中断
        return _rule_based_fallback(note)

# ---- Demo：Notebook/CLI 直接用 ----
def run_demo(text: str, use_model: bool = True):
    original = (text or "").strip()
    rewritten = rewrite_note(original, use_model=use_model)
    print("📝 Original:", original)
    print("✨ Rewritten:", rewritten)
    return {"original": original, "rewritten": rewritten}
