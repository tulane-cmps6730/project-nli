from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm
from .. import clf_path

import pickle
import sys


from flask import render_template, flash, redirect, session, request
from . import app
from .forms import MyForm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Assuming the model and tokenizer are loaded similarly to your Jupyter notebook
model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

label_names = ["entailment", "neutral", "contradiction"]

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = MyForm()
    result = None
    if form.validate_on_submit():
        premise = form.premise_field.data
        hypothesis = form.hypothesis_field.data
        print('premise: %s' % premise)
        print('hypothesis: %s' % hypothesis)

        # Tokenizing input
        inputs = tokenizer(premise, hypothesis, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs)
            confidence = probs[0, pred_idx].item()

        prediction = label_names[pred_idx]

        return render_template('myform.html', title='Interactive NLI Classifier', form=form, 
                               prediction=prediction, confidence='{:.2f}%'.format(confidence * 100))
    return render_template('myform.html', title='Interactive NLI Classifier', form=form, prediction=None, confidence=None)






# clf, vec = pickle.load(open(clf_path, 'rb'))
# print('read clf %s' % str(clf))
# print('read vec %s' % str(vec))
# labels = ['liberal', 'conservative']

# @app.route('/', methods=['GET', 'POST'])
# @app.route('/index', methods=['GET', 'POST'])
# def index():
# 	form = MyForm()
# 	result = None
# 	if form.validate_on_submit():
# 		premise = form.premise_field.data
# 		hypothesis = form.hypothesis_field.data
# 		print('premise: %s' % premise)
# 		print('hypothesis: %s' % hypothesis)
# 		X = vec.transform([input_field])
# 		pred = clf.predict(X)[0]
# 		proba = clf.predict_proba(X)[0].max()
# 		# flash(input_field)
# 		return render_template('myform.html', title='Interactive NLI Classifier', form=form, 
# 								prediction=labels[pred], confidence='%.2f' % proba)
# 		#return redirect('/index')
# 	return render_template('myform.html', title='Interactive NLI Classifier', form=form, prediction=None, confidence=None)


