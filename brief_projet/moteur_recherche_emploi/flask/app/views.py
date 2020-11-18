#!/usr/bin/env python
# coding: utf-8
from app import app
from app.models import *
from flask import render_template, request
from pymongo import MongoClient

client = MongoClient("localhost", 27017)
db = client["worksearch"]
CollectionMaongodb = db["search"]

@app.route('/', methods = ['GET', 'POST'])
def index():
    """Get the root with method """
    request.method == "POST"
    result = request.form.get('job')
    data = CollectionMaongodb.find( {"query": result} )
    #print(result)
    return render_template('index.html', job = data)