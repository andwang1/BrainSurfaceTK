from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def main():
    return "Welcome to SkyNet!"

@app.route("/about")
def about():
    return render_template('basic.html')

@app.route("/load_page")
def load():
    return render_template('load_page.html')

# "<h1 style='color: red;'>I'm a red H1 heading!</h1>"

if __name__ == "__main__":
    # http://localhost:5000/
    app.run(debug=True)
    # debug=True, host="0.0.0.0", port=80
