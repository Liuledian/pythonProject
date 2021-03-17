import time
from flask import Flask, request
import hashlib

print(__name__)
app = Flask(__name__)

SECRET = 'mac'


def verify(sig):
    if len(SECRET) != len(sig):
        return False
    for c1, c2 in zip(SECRET, sig):
        if c1 != c2:
            return False
        time.sleep(0.01)
    return True


def dummy_verify(sig):
    flag = True
    dummy = True
    for i in range(len(sig)):
        if i < len(SECRET):
            c1 = SECRET[i]
        else:
            c1 = None
        c2 = sig[i]
        if c1 != c2:
            flag = False
        else:
            # dummy operation
            dummy = flag
        time.sleep(0.01)
    return flag


def hash_verify(sig):
    h1 = hashlib.sha256()
    h1.update(sig)
    hash_value1 = h1.hexdigest()
    h2 = hashlib.sha256()
    h2.update(SECRET)
    hash_value2 = h2.hexdigest()
    for c1, c2 in zip(hash_value1, hash_value2):
        if c1 != c2:
            return False
        time.sleep(0.01)
    return True


@app.route("/naive")
def naive():
    token = request.headers.get('X-TOKEN')
    if not token:
        return "Missing token", 401
    if verify(token):
        return "CONGRATS, CORRECT TOKEN", 200
    else:
        return "ALERT, WRONG TOKEN", 403


@app.route("/dummy")
def standard():
    token = request.headers.get('X-TOKEN')
    if not token:
        return "Missing token", 401
    if dummy_verify(token):
        return "CONGRATS, CORRECT TOKEN", 200
    else:
        return "ALERT, WRONG TOKEN", 403


@app.route("/hash")
def mine():
    token = request.headers.get('X-TOKEN')
    if not token:
        return "Missing token", 401
    if hash_verify(token):
        return "CONGRATS, CORRECT TOKEN", 200
    else:
        return "ALERT, WRONG TOKEN", 403


if __name__ == "__main__":
    app.run()
