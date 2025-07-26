from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

@app.route("/", methods=["GET"])
def index():
    return "Embedding service running"

@app.route("/embed", methods=["POST"])
def embed():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = data["text"]
    if not isinstance(text, str):
        return jsonify({"error": "'text' must be a string"}), 400

    try:
        embedding = model.encode([text])[0].tolist()
        return jsonify({"embedding": embedding})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
