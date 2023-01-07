from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/req', methods=['POST'])
def add_message():
    content = request.json
    print(content)
    return "Ok"


if __name__ == '__main__':
    app.run(host='0.0.0.0')
