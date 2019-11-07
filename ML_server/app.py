from flask import Flask
from flask import request, Response
from flask_cors import CORS

import numpy as np
import re
import json

from flask import render_template
from ToxicCommentPredictor import ToxicCommentPredictor
toxic = ToxicCommentPredictor()

application = Flask(__name__)
CORS(application, supports_credentials=True)

@application.route("/")
def index():
    return 'sever is running now'

@application.route("/homepage")
def test():
    return render_template('demo.html')

@application.route('/mlserver', methods=['POST', 'GET'])
def mlserver():
    if request.form['key'] == 'dashuaige886':
        input = request.form['comment_text']
        # labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        output_array = toxic.rnn_prediction(input)
        return Response(json.dumps({
            'toxic':str(output_array[0]),
            'severe_toxic':str(output_array[1]), 
            'obscene':str(output_array[2]), 
            'threat':str(output_array[3]), 
            'insult':str(output_array[4]), 
            'identity_hate':str(output_array[5])
            }), mimetype='application/json')
    return Response('error',mimetype='text/xml')

if __name__ == "__main__":
    application.run(host='0.0.0.0', port='8080',threaded=False)
