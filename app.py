from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

@app.route("/embed", methods=["POST"])
def embed():
    data = request.json
    sentences = data.get("sentences", [])
    if not isinstance(sentences, list):
        return jsonify({"error": "Invalid input"}), 400

    embeddings = model.encode(sentences).tolist()
    return jsonify({"embeddings": embeddings})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))  # 👈 WAJIB
    app.run(host="0.0.0.0", port=port)
