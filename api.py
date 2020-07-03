from flask import Flask, request, jsonify
import models_runner

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Distant Reading Archive</h1>
<p>A prototype API for distant reading of science fiction novels.</p>'''


@app.route('/generate-text', methods=['GET'])
def api_all():
    query_parameters = request.args
    text = models_runner.generate_text(query_parameters.get('seed'),query_parameters.get('book'))
    output = {'text':text}
    return output

if __name__ == '__main__':
	app.run(debug=True)