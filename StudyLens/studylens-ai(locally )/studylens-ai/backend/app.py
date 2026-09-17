"""
StudyLens AI - Backend (نسخة محلية بالكامل عبر Ollama)
--------------------------------------------------------
خادم Flask بسيط يقوم بدورين:
1. تقديم ملفات الواجهة الأمامية (frontend/).
2. توسيط الطلبات إلى Ollama العامل محلياً على نفس الجهاز (http://localhost:11434).

لا حاجة لأي مفتاح API ولا اتصال إنترنت بعد تحميل الموديل — كل شيء يعمل على جهازك.

للتشغيل:
    1) ثبّت Ollama من https://ollama.com/download وشغّله.
    2) حمّل موديل الرؤية:  ollama pull qwen2.5vl:7b
    3) pip install -r requirements.txt
    4) python app.py
"""

import os
import re
import json
import requests
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip().rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:7b").strip()
PORT = int(os.environ.get("PORT", 5000))
# التوليد المحلي قد يكون بطيئاً (خصوصاً بدون GPU)، لذا نعطيه وقتاً كافياً
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", 300))

OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """أنت StudyLens AI، مساعد دراسي خبير. ستستلم صورة لصفحة تعليمية
(كتاب، محاضرة، رسم بياني، معادلة، جدول، أو مخطط). حلّل الصورة بدقة وأعد كائن JSON
يطابق المخطط المطلوب تماماً. اكتب كل النصوص بنفس لغة محتوى الصورة؛ إن كانت اللغة
غير واضحة أو مختلطة استخدم العربية. كن مختصراً ومفيداً:
- summary: 2-3 جمل فقط.
- key_concepts: 4 إلى 6 عناصر قصيرة.
- important_terms: 3 إلى 6 عناصر (term + definition قصير جداً).
- explanation: شرح واضح بحدود 120 كلمة يساعد طالباً على الفهم الفعلي.
- questions: 4 إلى 5 أسئلة مراجعة بدون إجابات.
"""

CHAT_SYSTEM_PROMPT = """أنت StudyLens AI، مدرّس خصوصي ودود وخبير يساعد طالباً على فهم صورة
صفحة تعليمية قام برفعها سابقاً (وتم تحليلها بالفعل ضمن هذه المحادثة). تابع الحوار
بشكل طبيعي معتمداً على الصورة الأصلية وسياق الرسائل السابقة. أجب دائماً بنفس لغة
سؤال الطالب (العربية افتراضياً). كن واضحاً ومنظماً (فقرات قصيرة أو نقاط عند الحاجة)
واختصر إجابتك ما أمكن دون فقدان الفائدة (بحدود 130 كلمة عادةً). إن طُلب منك اختبار
الطالب: اطرح سؤالاً واحداً فقط في كل رد، ثم قيّم إجابته بوضوح في ردك التالي قبل
الانتقال لسؤال جديد. إن طُلب شرح رسم/مخطط، صِف ما تراه بصرياً واربطه بالمفهوم.
إن طُلب تحويل جدول إلى مقارنة، قدّمها كنقاط منظمة وواضحة."""

ANALYSIS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "content_type": {
            "type": "string",
            "enum": ["book", "lecture", "chart", "equation", "table", "diagram", "other"],
        },
        "summary": {"type": "string"},
        "key_concepts": {"type": "array", "items": {"type": "string"}},
        "important_terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "definition": {"type": "string"},
                },
                "required": ["term", "definition"],
            },
        },
        "explanation": {"type": "string"},
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["content_type", "summary", "key_concepts", "important_terms", "explanation", "questions"],
}


class OllamaError(Exception):
    """Raised whenever the local Ollama call fails, with a message safe to show the user."""


def call_ollama(system_text, messages, json_schema=None):
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_text}] + messages,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 900},
    }
    if json_schema:
        payload["format"] = json_schema

    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    except requests.exceptions.ConnectionError:
        raise OllamaError(
            "تعذّر الاتصال بـ Ollama على جهازك. تأكد أنه يعمل (شغّل الأمر: ollama serve أو افتح تطبيق Ollama) "
            "ثم أعد المحاولة."
        )
    except requests.exceptions.Timeout:
        raise OllamaError("استغرق النموذج المحلي وقتاً طويلاً جداً للرد. جرّب موديلاً أصغر أو صورة أصغر.")
    except requests.exceptions.RequestException:
        raise OllamaError("حدث خطأ أثناء الاتصال بـ Ollama. حاول مرة أخرى.")

    if resp.status_code == 404:
        raise OllamaError(
            f"الموديل \"{MODEL}\" غير محمّل على جهازك. نزّله بالأمر: ollama pull {MODEL}"
        )
    if not resp.ok:
        try:
            err_msg = resp.json().get("error", "")
        except Exception:
            err_msg = ""
        raise OllamaError(f"خطأ من Ollama (كود {resp.status_code}): {err_msg}")

    data = resp.json()
    text = (data.get("message") or {}).get("content", "")
    if not text:
        raise OllamaError("رد فارغ من النموذج المحلي، حاول مرة أخرى.")
    return text


def clean_json_text(text):
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(silent=True) or {}
    image_b64 = body.get("image_base64")

    if not image_b64:
        return jsonify({"ok": False, "error": "لم يتم إرسال أي صورة."}), 400

    messages = [
        {"role": "user", "content": "حلّل هذه الصورة التعليمية.", "images": [image_b64]}
    ]

    try:
        raw_text = call_ollama(ANALYSIS_SYSTEM_PROMPT, messages, json_schema=ANALYSIS_RESPONSE_SCHEMA)
        parsed = json.loads(clean_json_text(raw_text))
    except OllamaError as e:
        return jsonify({"ok": False, "error": str(e)}), 502
    except json.JSONDecodeError:
        return jsonify({"ok": False, "error": "تعذّر فهم رد النموذج المحلي، حاول مرة أخرى."}), 502

    return jsonify({"ok": True, "data": parsed, "raw_text": raw_text})


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    history = body.get("history")

    if not history or not isinstance(history, list):
        return jsonify({"ok": False, "error": "لا يوجد سياق محادثة صالح."}), 400

    try:
        reply = call_ollama(CHAT_SYSTEM_PROMPT, history)
    except OllamaError as e:
        return jsonify({"ok": False, "error": str(e)}), 502

    return jsonify({"ok": True, "text": reply})


@app.route("/api/health")
def health():
    ollama_reachable = False
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        ollama_reachable = r.ok
    except requests.exceptions.RequestException:
        ollama_reachable = False
    return jsonify({"ok": True, "model": MODEL, "ollama_reachable": ollama_reachable})


if __name__ == "__main__":
    print(f"🧠 الموديل المحلي المستخدم: {MODEL}  (غيّره عبر OLLAMA_MODEL في backend/.env)")
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        print(f"✅ تم العثور على Ollama يعمل على {OLLAMA_HOST}")
    except requests.exceptions.RequestException:
        print(f"\n⚠️  تحذير: لا يمكن الوصول إلى Ollama على {OLLAMA_HOST}")
        print("   ثبّت Ollama من https://ollama.com/download وشغّله، ثم نزّل الموديل بالأمر:")
        print(f"   ollama pull {MODEL}\n")
    print(f"🚀 StudyLens AI يعمل الآن على: http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=True)
