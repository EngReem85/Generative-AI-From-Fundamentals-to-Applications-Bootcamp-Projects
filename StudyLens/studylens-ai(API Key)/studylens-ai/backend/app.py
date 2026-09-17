"""
StudyLens AI - Backend
-----------------------
خادم Flask بسيط يقوم بدورين:
1. تقديم ملفات الواجهة الأمامية (frontend/).
2. توسيط الطلبات إلى Gemini API المجاني (Google AI Studio) حتى لا يظهر
   مفتاح الـ API في متصفح المستخدم.

للتشغيل:
    pip install -r requirements.txt
    python app.py
"""

import os
import re
import json
import requests
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
PORT = int(os.environ.get("PORT", 5000))

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

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
    "type": "OBJECT",
    "properties": {
        "content_type": {
            "type": "STRING",
            "enum": ["book", "lecture", "chart", "equation", "table", "diagram", "other"],
        },
        "summary": {"type": "STRING"},
        "key_concepts": {"type": "ARRAY", "items": {"type": "STRING"}},
        "important_terms": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "term": {"type": "STRING"},
                    "definition": {"type": "STRING"},
                },
                "required": ["term", "definition"],
            },
        },
        "explanation": {"type": "STRING"},
        "questions": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["content_type", "summary", "key_concepts", "important_terms", "explanation", "questions"],
}


class GeminiError(Exception):
    """Raised whenever the Gemini API call fails, with a message safe to show the user."""


def call_gemini(system_text, contents, use_schema=False):
    if not API_KEY:
        raise GeminiError(
            "لم يتم ضبط مفتاح Gemini API. أضف GEMINI_API_KEY داخل ملف backend/.env ثم أعد تشغيل الخادم."
        )

    generation_config = {"maxOutputTokens": 1200, "temperature": 0.4}
    if use_schema:
        generation_config["response_mime_type"] = "application/json"
        generation_config["response_schema"] = ANALYSIS_RESPONSE_SCHEMA

    payload = {
        "system_instruction": {"parts": [{"text": system_text}]},
        "contents": contents,
        "generationConfig": generation_config,
    }

    try:
        resp = requests.post(
            GEMINI_URL,
            params={"key": API_KEY},
            json=payload,
            timeout=60,
        )
    except requests.exceptions.RequestException:
        raise GeminiError("تعذّر الاتصال بخدمة Gemini. تحقق من اتصالك بالإنترنت وحاول مرة أخرى.")

    if resp.status_code == 429:
        raise GeminiError("تم تجاوز الحد المجاني المسموح به مؤقتاً. انتظر دقيقة ثم حاول مرة أخرى.")
    if resp.status_code == 400:
        raise GeminiError("طلب غير صالح إلى Gemini، تحقق من صحة مفتاح الـ API واسم الموديل في .env.")
    if resp.status_code in (401, 403):
        raise GeminiError("مفتاح Gemini API غير صالح أو غير مفعّل. تحقق منه في backend/.env.")
    if not resp.ok:
        raise GeminiError(f"خطأ غير متوقع من خدمة Gemini (كود {resp.status_code}).")

    data = resp.json()

    candidates = data.get("candidates") or []
    if not candidates:
        feedback = data.get("promptFeedback", {})
        raise GeminiError(f"لم يُرجع النموذج أي رد. ({feedback.get('blockReason', 'سبب غير معروف')})")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    if not text:
        raise GeminiError("رد فارغ من النموذج، حاول مرة أخرى.")
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
    mime_type = body.get("mime_type", "image/jpeg")

    if not image_b64:
        return jsonify({"ok": False, "error": "لم يتم إرسال أي صورة."}), 400

    contents = [
        {
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                {"text": "حلّل هذه الصورة التعليمية."},
            ],
        }
    ]

    try:
        raw_text = call_gemini(ANALYSIS_SYSTEM_PROMPT, contents, use_schema=True)
        parsed = json.loads(clean_json_text(raw_text))
    except GeminiError as e:
        return jsonify({"ok": False, "error": str(e)}), 502
    except json.JSONDecodeError:
        return jsonify({"ok": False, "error": "تعذّر فهم رد النموذج، حاول مرة أخرى."}), 502

    return jsonify({"ok": True, "data": parsed, "raw_text": raw_text})


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    history = body.get("history")

    if not history or not isinstance(history, list):
        return jsonify({"ok": False, "error": "لا يوجد سياق محادثة صالح."}), 400

    try:
        reply = call_gemini(CHAT_SYSTEM_PROMPT, history)
    except GeminiError as e:
        return jsonify({"ok": False, "error": str(e)}), 502

    return jsonify({"ok": True, "text": reply})


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "model": MODEL, "key_configured": bool(API_KEY)})


if __name__ == "__main__":
    if not API_KEY:
        print("\n⚠️  تحذير: GEMINI_API_KEY غير موجود في backend/.env")
        print("   انسخ backend/.env.example إلى backend/.env وضع مفتاحك المجاني من:")
        print("   https://aistudio.google.com/apikey\n")
    print(f"🚀 StudyLens AI يعمل الآن على: http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=True)
