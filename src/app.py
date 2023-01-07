import json

from flask import Flask, request

from src.services.checker_service import check_and_transform
from src.structs.request_structs import CheckerRequestFormat, MyEncoder

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!\n To use this app send POST request on /req with json of template\n ' \
           '{\"text\":\"Текст на русском, где потребуются запятые и прочие знаки препинания.\"}'


@app.route('/req', methods=['POST'])
def add_message():
    content = request.json
    print(content)
    req = CheckerRequestFormat(content["text"])
    req = check_and_transform(req)
    return json.dumps(req, cls=MyEncoder)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
