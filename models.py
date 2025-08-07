from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    entry_time = db.Column(db.String(50))
    exit_time = db.Column(db.String(50))
    symbol = db.Column(db.String(20))
    direction = db.Column(db.String(10))
    entry_price = db.Column(db.Float)
    exit_price = db.Column(db.Float)
    pnl = db.Column(db.Float)
    capital = db.Column(db.Float)

class EquityLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(50))
    equity = db.Column(db.Float)
