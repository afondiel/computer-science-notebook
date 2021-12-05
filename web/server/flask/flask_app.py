from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "test flask"

if __name__ == '__main__':
    # print('test')
    app.run()