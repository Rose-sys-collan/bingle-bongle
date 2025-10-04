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


# ========== Demo helper functions (EchoClass) ==========
# ---- 核心：调用模型的改写函数（支持 use_model 参数）----
import os, re

def _rule_fallback(text: str) -> str:
    s = (text or "").strip()
    pairs = [
        (r"\bforeign students\b", "international students"),
        (r"\bstudents with (strong|thick) accents\b", "students who speak with diverse accents"),
        (r"\bpoor English\b", "English in progress"),
        (r"\bbad English\b", "English in progress"),
        (r"\btoo fast\b", "a bit fast"),
        (r"\bhard words\b", "complex words"),
        (r"\bstruggle to present\b", "may face challenges presenting"),
    ]
    for a, b in pairs:
        s = re.sub(a, b, s, flags=re.IGNORECASE)
    if re.search(r"\baccents\b", s, re.I) and re.search(r"\bpresent", s, re.I):
        if not re.search(r"can .*present .*effectively", s, re.I):
            s = s.rstrip(".") + ", and with inclusive feedback and practice, they can present effectively."
    return re.sub(r"\s+", " ", s).strip() or "This note has been rewritten in clear and inclusive English."

# 如果你上面已经 import 过 google.generativeai as genai，就不用再 import；否则解除注释
# import google.generativeai as genai

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
""".strip()

def rewrite_note(
    text: str,
    use_model: bool = True,
    model_name: str | None = None,
    temperature: float = 0.4,
) -> str:
    """Rewrite with Gemini when available; otherwise fall back to simple rules."""
    note = (text or "").strip()
    if not note:
        return ""

    if not use_model:
        return _rule_fallback(note)

    # 读取环境变量中的模型名（若未传入）
    model_name = model_name or os.getenv("ECHOCLASS_MODEL", "gemini-1.5-flash")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    # 如果没 key 或 SDK 不可用，就退回规则版
    try:
        import google.generativeai as genai
    except Exception:
        return _rule_fallback(note)
    if not api_key:
        return _rule_fallback(note)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = PROMPT_NOTE_REWRITE.format(note=note)
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": float(temperature), "top_p": 0.95, "top_k": 40},
        )
        out = (getattr(resp, "text", "") or "").strip()
        # 清理可能的说明性前缀
        out = re.sub(r"^(Rewrite[d]?|Rewriting|Here\s+is|Output)\s*[:\-–]\s*", "", out, flags=re.I)
        out = re.sub(r"\s+", " ", out).strip()
        return out or _rule_fallback(note)
    except Exception:
        return _rule_fallback(note)


def run_demo(text: str, use_model: bool = True):
    """
    Quick demo for Jupyter or CLI.
    If use_model=True, try Gemini; otherwise fall back to rule-based.
    """
    original = (text or "").strip()
    rewritten = rewrite_note(original, use_model=use_model)
    print("📝 Original:", original)
    print("✨ Rewritten:", rewritten)
    return {"original": original, "rewritten": rewritten}

# ========== Explain Terms in Plain English ==========

import re, os
from typing import List, Dict, Optional

# 轻量、可扩展的内置词表（fallback 用；可改成 docs/glossary.json）
_FALLBACK_GLOSSARY = {
    "GPA": "Grade Point Average, your overall score in school on a numeric scale.",
    "TA": "Teaching Assistant, a student who helps the professor with teaching and grading.",
    "midterm": "An exam in the middle of a course to check progress.",
    "final": "The last big exam at the end of a course.",
    "syllabus": "A document that explains what a course covers and how you will be graded.",
    "citation": "A reference to the source of information you used.",
    "plagiarism": "Using someone else’s words or ideas as your own without credit.",
    "rubric": "A scoring guide that shows how your work will be graded.",
}

# 抓术语/缩略词/专有名词（简单启发式）
def extract_terms(text: str, max_terms: int = 8) -> List[str]:
    s = text or ""
    # 1) 缩略词：2~6 个大写字母
    acronyms = re.findall(r"\b[A-Z]{2,6}\b", s)
    # 2) 驼峰/首字母大写短语（最多 3 词）
    caps_phrases = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", s)
    # 3) 关键术语（含连字符/后缀）
    hyphen_terms = re.findall(r"\b[a-zA-Z]+(?:-[a-zA-Z]+)+\b", s)

    # 合并去重，过滤常见词
    raw = acronyms + caps_phrases + hyphen_terms
    seen, out = set(), []
    stop = {"The","This","That","And","Or","But","We","You","They","He","She","It","A","An","In","On","At","For","Of","To","From","By"}
    for t in raw:
        tok = t.strip()
        if tok in stop: 
            continue
        low = tok.lower()
        if low not in seen and len(tok) >= 2:
            seen.add(low)
            out.append(tok)
    return out[:max_terms]

# Prompt（要求 CEFR A2-B1，禁术语化）
PROMPT_EXPLAIN_TERMS = """
You are EchoClass. Explain academic terms in plain English for ESL students (CEFR A2–B1).
Rules:
- Use 1–2 short sentences per term.
- Avoid jargon; use simple, concrete words.
- Include an everyday example if helpful.
- Be neutral and inclusive.
- Output JSON list of objects: [{"term": "...","explanation": "..."}] with only the JSON.

Terms:
{terms}
Context:
{context}
""".strip()

def _simple_explain(term: str) -> str:
    # 先查内置词表，再给通用解释
    if term.upper() in _FALLBACK_GLOSSARY:
        return _FALLBACK_GLOSSARY[term.upper()]
    if term.lower() in _FALLBACK_GLOSSARY:
        return _FALLBACK_GLOSSARY[term.lower()]
    # 通用模板（避免完全空）
    return f"{term} means something used in this class. In simple words, it helps you study or understand the topic."

def explain_terms(
    text: str,
    terms: Optional[List[str]] = None,
    use_model: bool = True,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
) -> List[Dict[str, str]]:
    """
    Return a list of {"term": str, "explanation": str}.
    Tries Gemini first; falls back to simple dictionary rules.
    """
    ctx = (text or "").strip()
    terms = terms or extract_terms(ctx)
    if not terms:
        return []

    if not use_model:
        return [{"term": t, "explanation": _simple_explain(t)} for t in terms]

    # 读取环境与依赖
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = model_name or os.getenv("ECHOCLASS_MODEL", "gemini-1.5-flash")
    try:
        import google.generativeai as genai
        if not api_key:
            raise RuntimeError("No API key")
        genai.configure(api_key=api_key)
        prompt = PROMPT_EXPLAIN_TERMS.format(
            terms=", ".join(terms),
            context=ctx[:1200]  # 防止超长
        )
        resp = genai.GenerativeModel(model_name).generate_content(
            prompt,
            generation_config={"temperature": float(temperature), "top_p": 0.9, "top_k": 40},
        )
        raw = (getattr(resp, "text", "") or "").strip()
        # 安全解析 JSON
        import json
        # 简单清洗：去掉模型可能加的前后文
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(raw)
        # 兜底：结构不齐时修补
        out = []
        for t in terms:
            hit = next((d for d in data if isinstance(d, dict) and d.get("term", "").strip().lower()==t.strip().lower()), None)
            exp = hit.get("explanation") if hit else None
            out.append({"term": t, "explanation": exp or _simple_explain(t)})
        return out
    except Exception:
        # 任意异常：完全回退
        return [{"term": t, "explanation": _simple_explain(t)} for t in terms]
# ========== Explain Terms in Plain English ==========

import re, os
from typing import List, Dict, Optional

# 轻量、可扩展的内置词表（fallback 用；可改成 docs/glossary.json）
_FALLBACK_GLOSSARY = {
    "GPA": "Grade Point Average, your overall score in school on a numeric scale.",
    "TA": "Teaching Assistant, a student who helps the professor with teaching and grading.",
    "midterm": "An exam in the middle of a course to check progress.",
    "final": "The last big exam at the end of a course.",
    "syllabus": "A document that explains what a course covers and how you will be graded.",
    "citation": "A reference to the source of information you used.",
    "plagiarism": "Using someone else’s words or ideas as your own without credit.",
    "rubric": "A scoring guide that shows how your work will be graded.",
}

# 抓术语/缩略词/专有名词（简单启发式）
def extract_terms(text: str, max_terms: int = 8) -> List[str]:
    s = text or ""
    # 1) 缩略词：2~6 个大写字母
    acronyms = re.findall(r"\b[A-Z]{2,6}\b", s)
    # 2) 驼峰/首字母大写短语（最多 3 词）
    caps_phrases = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", s)
    # 3) 关键术语（含连字符/后缀）
    hyphen_terms = re.findall(r"\b[a-zA-Z]+(?:-[a-zA-Z]+)+\b", s)

    # 合并去重，过滤常见词
    raw = acronyms + caps_phrases + hyphen_terms
    seen, out = set(), []
    stop = {"The","This","That","And","Or","But","We","You","They","He","She","It","A","An","In","On","At","For","Of","To","From","By"}
    for t in raw:
        tok = t.strip()
        if tok in stop: 
            continue
        low = tok.lower()
        if low not in seen and len(tok) >= 2:
            seen.add(low)
            out.append(tok)
    return out[:max_terms]

# Prompt（要求 CEFR A2-B1，禁术语化）
PROMPT_EXPLAIN_TERMS = """
You are EchoClass. Explain academic terms in plain English for ESL students (CEFR A2–B1).
Rules:
- Use 1–2 short sentences per term.
- Avoid jargon; use simple, concrete words.
- Include an everyday example if helpful.
- Be neutral and inclusive.
- Output JSON list of objects: [{"term": "...","explanation": "..."}] with only the JSON.

Terms:
{terms}
Context:
{context}
""".strip()

def _simple_explain(term: str) -> str:
    # 先查内置词表，再给通用解释
    if term.upper() in _FALLBACK_GLOSSARY:
        return _FALLBACK_GLOSSARY[term.upper()]
    if term.lower() in _FALLBACK_GLOSSARY:
        return _FALLBACK_GLOSSARY[term.lower()]
    # 通用模板（避免完全空）
    return f"{term} means something used in this class. In simple words, it helps you study or understand the topic."

def explain_terms(
    text: str,
    terms: Optional[List[str]] = None,
    use_model: bool = True,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
) -> List[Dict[str, str]]:
    """
    Return a list of {"term": str, "explanation": str}.
    Tries Gemini first; falls back to simple dictionary rules.
    """
    ctx = (text or "").strip()
    terms = terms or extract_terms(ctx)
    if not terms:
        return []

    if not use_model:
        return [{"term": t, "explanation": _simple_explain(t)} for t in terms]

    # 读取环境与依赖
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = model_name or os.getenv("ECHOCLASS_MODEL", "gemini-1.5-flash")
    try:
        import google.generativeai as genai
        if not api_key:
            raise RuntimeError("No API key")
        genai.configure(api_key=api_key)
        prompt = PROMPT_EXPLAIN_TERMS.format(
            terms=", ".join(terms),
            context=ctx[:1200]  # 防止超长
        )
        resp = genai.GenerativeModel(model_name).generate_content(
            prompt,
            generation_config={"temperature": float(temperature), "top_p": 0.9, "top_k": 40},
        )
        raw = (getattr(resp, "text", "") or "").strip()
        # 安全解析 JSON
        import json
        # 简单清洗：去掉模型可能加的前后文
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(raw)
        # 兜底：结构不齐时修补
        out = []
        for t in terms:
            # 找到对应项
            hit = next((d for d in data if isinstance(d, dict) and d.get("term", "").strip().lower()==t.strip().lower()), None)
            exp = hit.get("explanation") if hit else None
            out.append({"term": t, "explanation": exp or _simple_explain(t)})
        return out
    except Exception:
        # 任意异常：完全回退
        return [{"term": t, "explanation": _simple_explain(t)} for t in terms]


# ========== Elaborate Based on Tone ==========

PROMPT_ELABORATE_TONE = """
You are EchoClass. Elaborate the idea for ESL students in {audience} with the requested tone.
Requirements:
- Tone: {tone}
- Length: {length_hint}
- Level: CEFR A2–B1 (plain, concrete English)
- Be inclusive and non-judgmental.
- Avoid jargon unless explained.
Return only the rewritten text (no preface).
Original:
{original}
""".strip()

def elaborate_note(
    text: str,
    tone: str = "friendly and encouraging",
    audience: str = "undergraduate students",
    length: str = "short",
    use_model: bool = True,
    model_name: str | None = None,
    temperature: float = 0.4,
) -> str:
    original = (text or "").strip()
    if not original:
        return ""
    length_map = {
        "very short": "2–3 sentences",
        "short": "4–6 sentences",
        "medium": "1–2 paragraphs",
        "long": "2–3 paragraphs",
    }
    length_hint = length_map.get(length, "4–6 sentences")
    if not use_model:
        base = original.rstrip(".")
        return f"{base}. In simple words, {base.lower()}. This version is {tone} for {audience}."
    import os
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = model_name or os.getenv("ECHOCLASS_MODEL", "gemini-1.5-flash")
    try:
        import google.generativeai as genai
        if not api_key:
            raise RuntimeError("No API key")
        genai.configure(api_key=api_key)
        prompt = PROMPT_ELABORATE_TONE.format(
            audience=audience, tone=tone, length_hint=length_hint, original=original
        )
        resp = genai.GenerativeModel(model_name).generate_content(
            prompt, generation_config={"temperature": float(temperature), "top_p": 0.9, "top_k": 40}
        )
        return (getattr(resp, "text", "") or "").strip()
    except Exception:
        base = original.rstrip(".")
        return f"{base}. In simple words, {base.lower()}. This version is {tone} for {audience}."

# ========== Local Source Retrieval (TF-IDF over docs/) ==========

from pathlib import Path
from typing import List, Dict, Optional
import re

def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return p.read_text(errors="ignore")

def _read_pdf_file(p: Path) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(p))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""

def _make_title(p: Path) -> str:
    return p.stem.replace("_"," ").replace("-"," ").title()

def _make_snippet(text: str, query: str, max_len: int = 200) -> str:
    q = re.escape(query.split()[0]) if query.strip() else ""
    if q:
        m = re.search(q, text, flags=re.I)
        if m:
            i = max(0, m.start() - 80)
            return (text[i:i+max_len].replace("\n", " ")).strip()
    return (text[:max_len].replace("\n", " ")).strip()

def retrieve_sources(
    query: str,
    root: str | Path = "docs",
    patterns: tuple = ("*.md","*.txt","*.pdf"),
    top_k: int = 3,
) -> List[Dict]:
    root = Path(root)
    files = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    if not files:
        return []
    corpus_paths, corpus_texts = [], []
    for p in files:
        txt = _read_pdf_file(p) if p.suffix.lower()==".pdf" else _read_text_file(p)
        txt = (txt or "").strip()
        if txt:
            corpus_paths.append(p)
            corpus_texts.append(txt)
    if not corpus_texts:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
        X = vect.fit_transform(corpus_texts)
        qv = vect.transform([query])
        sims = cosine_similarity(qv, X).ravel()
        idxs = sims.argsort()[::-1][:top_k]
    except Exception:
        sims = [sum(t.lower().count(w) for w in query.lower().split()) for t in corpus_texts]
        idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
    out: List[Dict] = []
    for i in idxs:
        p, t, s = corpus_paths[i], corpus_texts[i], float(sims[i])
        out.append({
            "path": str(p),
            "title": _make_title(p),
            "snippet": _make_snippet(t, query),
            "score": round(s, 4),
        })
    return out
