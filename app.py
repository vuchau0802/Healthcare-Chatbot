from dotenv import load_dotenv
load_dotenv()
def preprocess(text):
    return text
from flask import Flask, render_template, request, jsonify, session
from agents.orchestrator import ask_medical_question

import os
import uuid
import logging

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", uuid.uuid4().hex)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants
MAX_QUESTION_LENGTH = 1000
MAX_HISTORY_TURNS   = 6


@app.route("/")
def home():
    session.setdefault("history", [])
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True)

        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON body."}), 400

        question = data.get("message", "").strip()

        if not question:
            return jsonify({"error": "Message cannot be empty."}), 400

        if len(question) > MAX_QUESTION_LENGTH:
            return jsonify({
                "error": f"Message too long. Please keep it under {MAX_QUESTION_LENGTH} characters."
            }), 400

        history: list[dict] = session.get("history", [])

        logger.info("Question [session=%s]: %s", session.get("sid", "new"), question[:80])

        answer = ask_medical_question(question, history=history)

        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": answer})
        session["history"] = history[-(MAX_HISTORY_TURNS * 2):]

        if "sid" not in session:
            session["sid"] = uuid.uuid4().hex[:8]

        logger.info("Answer [session=%s]: %s …", session["sid"], answer[:80])

        return jsonify({
            "response":   answer,
            "session_id": session["sid"],
        })

    except Exception:
        logger.exception("Unhandled error in /chat")
        return jsonify({
            "error": "Something went wrong on our end. Please try again."
        }), 500


@app.route("/reset", methods=["POST"])
def reset():
    session.pop("history", None)
    session.pop("sid", None)
    return jsonify({"status": "History cleared."})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "mode": "multi-agent"})


if __name__ == "__main__":
    app.run(debug=False, port=8080)
