# coding: utf8
from app import app

# importer les autres éléments déclarés 
# dans /app/__init__py selon les besoins
#
# from app import db, babel

# importer les modèles pour accéder 
# aux données
#
#from app.models import *
from flask import render_template, request
from app.function import loadData, recommendation


@app.route('/')
def index():
  artists_name = loadData()
  return render_template('index.html', artists_name = artists_name)

@app.route('/result', methods = ['GET', 'POST'])
def result():
    request.method == 'POST'
    name = request.form.getlist('artist')

    items = recommendation(name)
    
    return render_template('result.html', name = name, items = items)
