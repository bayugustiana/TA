from flask import Flask, url_for, jsonify, redirect, request, session
from flask_session import Session
from flask_mysqldb import MySQL

import torch
import torch.nn as nn
import numpy as np

from hyperparameters import device
from architecture import ANN
from data import sc_X, X_train, y_train, X_test, y_test
from predict import get_pred

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'mooid'
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

mysql = MySQL(app)
@app.route('/aktivitas', methods=['GET'])
def aktivitas():
 cur = mysql.connection.cursor()
 cur.execute('''select suhu, detak_jantung from aktivitas where id=1''')
 rv = cur.fetchall()
 session['rv'] = rv
 return redirect(url_for('predict'))

@app.route('/predict')
def predict():
    model = ANN(2).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    checkpoint = torch.load('model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    my_var = session.get('rv', None)
    
    my_var = np.array(my_var, dtype=np.float32).reshape(1, -1)
    
    my_var = sc_X.transform(my_var)
    pred = get_pred(torch.from_numpy(my_var).to(device))
    
    return jsonify(str(pred[0]))

if __name__ == '__main__':
    app.run()