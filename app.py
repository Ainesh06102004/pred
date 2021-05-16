from flask import Flask, request, jsonify
from prediction import make_prediction
app = Flask(__name__)

@app.route("/predict-digit", methods = ["POST"])
def predictdata():
    image = request.files.get("digit")
    prediction_result = make_prediction(image)
    return jsonify({
        "prediction made": prediction_result
    })

if(__name__ == "__main__"):
    app.run(debug=True)