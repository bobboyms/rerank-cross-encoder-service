from utils import decode
from flask import Flask, jsonify, request
from sentence_transformers import CrossEncoder

app = Flask(__name__)
model = CrossEncoder('./model')
model.max_seq_length = 256


@app.route('/re-rank', methods=['POST'])
def hello_geek():
    request_data = request.get_json()

    query = request_data['query']
    documents = request_data['documents']

    passages = []
    for document in documents:
        passages.append(document["text"])

    model_inputs = [[query, passage] for passage in passages]
    scores = model.predict(model_inputs)

    print(scores)

    documents_scores = []
    length = len(documents)
    for i in range(length):
        id_document = documents[i]["id"]
        score = decode(scores[i])
        documents_scores.append({"id": id_document, "score": score})

    return jsonify({
        "length": length,
        "scores": documents_scores
    }), 200, {'ContentType': 'application/json'}


if __name__ == "__main__":
    # app.run(host="localhost", port=8000, debug=True)
    app.run(debug=True)
