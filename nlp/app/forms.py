from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class MyForm(FlaskForm):
	class Meta:  # Ignoring CSRF security feature.
		csrf = False

	premise_field = StringField(label='Premise:', id='premise_field',
							  validators=[DataRequired()], 
							  render_kw={'style': 'width:50%'})
	hypothesis_field = StringField(label='Hypothesis:', id='hypo_field',
							  validators=[DataRequired()], 
							  render_kw={'style': 'width:50%'})
	submit = SubmitField('Submit')