from flask import Flask, request, url_for, redirect, render_template
import nltk
import summarisation
from nltk import word_tokenize
nltk.download('punkt')
import math
app = Flask(__name__, template_folder='templates')
from jinja2 import Environment

def enumerate_filter(iterable, start=0):
    return enumerate(iterable, start=start)


env = Environment()
env.filters['enumerate'] = enumerate_filter

@app.route('/')
def hello_world():
    return render_template("stimulation.html")

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    name = request.form['name']
    email = request.form['email']
    feedback = request.form['feedback']

    # Save the feedback data to a file
    with open('feedback.txt', 'a') as f:
        f.write(f'Name: {name}\nEmail: {email}\nFeedback: {feedback}\n\n')
        return render_template('c.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    input_text = request.form['input_text'].strip()
    rule_based = summarisation.rule_based_summarization(input_text).strip()
    abstractive = summarisation.abstractive_summarization(input_text).strip()
    extractive = summarisation.extractive_summarization(input_text).strip()
    graph_based = summarisation.graph_based_summarization(input_text).strip()
    data = { 'input': input_text,
        'extractive' : extractive,
             'abstractive' : abstractive,
             'graph_based' : graph_based,
             'rule_based' : rule_based}
    # Do something with the input_text
    return render_template('stimulation.html', data=data)

@app.route('/b')
def b():
    return render_template("stimulation.html")
@app.route('/a')
def a():
    return render_template("a.html")
@app.route('/c')
def c():
    return render_template("c.html")

@app.route('/x')
def x():
    return render_template("x.html")

if __name__ == '__main__':
    app.run(debug=True)
