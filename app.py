from flask import Flask, render_template, request, jsonify
from rag import ask_medical_question

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json()

        question = data["message"]

        print("Question:", question)

        response = ask_medical_question(question)

        print("Response:", response)

        return jsonify({
            "response": response
        })

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "response": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True, port=8080)