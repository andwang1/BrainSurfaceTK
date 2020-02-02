import os
import time
import math
from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions
from werkzeug.security import check_password_hash, generate_password_hash

from helpers import apology, login_required, lookup, usd

# Configure application
app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# Custom filter
app.jinja_env.filters["usd"] = usd

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///finance.db")


@app.route("/")
# @login_required
def index():
    holdings = db.execute("SELECT SUM(quantity) as quantity,ticker FROM transactions t, stocks s WHERE s.stockid = t.stockid AND t.userid = :userid GROUP BY ticker",userid = session["user_id"])
    stocksheld = [stock["ticker"] for stock in holdings if stock["quantity"] != 0]
    quantityheld = [stock["quantity"] for stock in holdings if stock["ticker"] in stocksheld]
    priceofstocks = [lookup(stock)["price"] for stock in stocksheld]
    valueheld = []
    for i in range(len(priceofstocks)):
        valueheld.append(quantityheld[i] * priceofstocks[i])
    currentcash = db.execute("SELECT cash from users where id = :userid", userid = session["user_id"])[0]["cash"]
    stockvalues = sum(valueheld)
    portfoliovalue = usd(currentcash + stockvalues)
    priceofstocks = [usd(price) for price in priceofstocks]
    valueheld = [usd(value) for value in valueheld]
    currentcash = usd(currentcash)
    dataset = list(zip(stocksheld,quantityheld,priceofstocks,valueheld))
    return render_template("index.html",data = dataset, currentcash = currentcash, portfoliovalue = portfoliovalue)


@app.route("/buy", methods=["GET", "POST"])
@login_required
def buy():
    if request.method == "GET":
        return render_template("buy.html")
    else:
        if not request.form.get("symbol"):
            return apology("must provide ticker", 400)
        if request.form.get("symbol") == "":
            return apology("must provide ticker", 400)
        if not lookup(request.form.get("symbol")):
            return apology("stock not found", 400)
        if not request.form.get("shares"):
            return apology("please input positive integer for number of shares", 400)
        if not request.form.get("shares").isnumeric():
            return apology("please input positive integer for number of shares", 400)
        if int(request.form.get("shares")) < 0:
            return apology("please input positive integer for number of shares", 400)
        if int(request.form.get("shares")) - math.floor(int(request.form.get("shares"))) != 0:
            return apology("please input positive integer for number of shares", 400)
        purchaseqty = int(request.form.get("shares"))
        stockdict = lookup(request.form.get("symbol"))
        stockname = stockdict["name"]
        stockprice = stockdict["price"]
        stockticker = stockdict["symbol"]
        purchasevalue = stockprice * purchaseqty
        usercash = db.execute("SELECT cash from users where id = :userid",userid = session["user_id"])[0]["cash"]
        if usercash < purchasevalue:
            return apology("not enough funds to cover purchase", 400)
        else:
            stockregistered = db.execute("SELECT * FROM stocks WHERE ticker = :ticker", ticker = stockticker)
            if len(stockregistered) != 1:
                db.execute("INSERT INTO stocks (name, ticker) VALUES (:name, :ticker)", name = stockname, ticker = stockticker)
            stockid = db.execute("SELECT stockid from stocks WHERE ticker = :ticker", ticker = stockticker)[0]["stockid"]
            db.execute("INSERT INTO transactions (userid, quantity, price, stockid, trantype, time) VALUES (:userid, :qty, :price, :stockid, 'BUY', :time)", userid = session["user_id"], qty = purchaseqty, price = stockprice, stockid = stockid, time = time.strftime('%Y-%m-%d %H:%M:%S'))
            db.execute("UPDATE users SET cash = :newcash WHERE id = :userid", newcash = usercash - purchasevalue, userid = session["user_id"])
            return redirect("/")


@app.route("/history")
@login_required
def history():
    transactions = db.execute("SELECT ticker, trantype, abs(quantity), price, time FROM stocks s, transactions t where s.stockid = t.stockid and t.userid = :userid", userid = session["user_id"])
    transactiontuples=[]
    for transaction in transactions:
        transactiontuples.append(tuple(transaction.values()))
    return render_template("history.html",data = transactiontuples)

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 400)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password", 400)

        # Query database for username
        rows = db.execute("SELECT * FROM users WHERE username = :username",
                          username=request.form.get("username"))

        # # Ensure username exists and password is correct
        # if len(rows) != 1 or not check_password_hash(rows[0]["hash"], request.form.get("password")):
        #     return apology("invalid username and/or password", 400)

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")


@app.route("/quote", methods=["GET", "POST"])
@login_required
def quote():
    if request.method == "GET":
        return render_template("quote.html")
    else:
        if not request.form.get("symbol"):
            return apology("must provide ticker", 400)
        if not lookup(request.form.get("symbol")):
            return apology("stock not found", 400)
        stockdict = lookup(request.form.get("symbol"))
        return render_template("quoted.html",stockname = stockdict["name"],stockprice = usd(stockdict["price"]),stockticker = stockdict["symbol"])


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        session.clear()
        if not request.form.get("username"):
            return apology("must provide username", 400)
        elif request.form.get("username") =="":
            return apology("must provide username", 400)
        elif not request.form.get("password"):
            return apology("must provide a password", 400)
        elif request.form.get("password") != request.form.get("confirmation"):
            return apology("password and confirmation do not match", 400)
        elif request.form.get("password").isalpha():
            return apology("password requires at least one non-alphabetical character", 400)
        else:
            newusername = request.form.get("username")
            rows = db.execute("SELECT username from users WHERE username = :user",user = request.form.get("username"))
            if len(rows) !=0:
                return apology("username already taken", 400)
            db.execute("INSERT INTO users (username, hash) VALUES(:user, :password)",user = request.form.get("username"), password = generate_password_hash(request.form.get("password")))
            return redirect("/login")
    else:
        return render_template("register.html")

@app.route("/sell", methods=["GET", "POST"])
@login_required
def sell():
    if request.method == "GET":
        holdings = db.execute("SELECT SUM(quantity) as quantity,ticker FROM transactions t, stocks s WHERE s.stockid = t.stockid AND t.userid = :userid GROUP BY ticker",userid = session["user_id"])
        stocksheld = [stock["ticker"] for stock in holdings if stock["quantity"] > 0]
        return render_template("sell.html", stocksheld = stocksheld)
    else:
        holdings = db.execute("SELECT SUM(quantity) as quantity,ticker FROM transactions t, stocks s WHERE s.stockid = t.stockid AND t.userid = :userid GROUP BY ticker",userid = session["user_id"])
        stocksheld = [stock["ticker"] for stock in holdings if stock["quantity"] > 0]
        if request.form.get("symbol") not in set(stocksheld):
            return apology("stock not held",400)
        elif not request.form.get("symbol"):
            return apology("please select a stock to sell",400)
        stocksold = request.form.get("symbol")
        quantitysold = int(request.form.get("shares"))
        for holding in holdings:
            if stocksold == holding["ticker"]:
                quantityowned = holding["quantity"]
                break
        if quantitysold > quantityowned:
            return apology("attempting to sell more stocks than owned", 400)
        else:
            stockdict = lookup(stocksold)
            stockprice = stockdict["price"]
            salevalue = stockprice * quantitysold
            usercash = db.execute("SELECT cash from users where id = :userid",userid = session["user_id"])[0]["cash"]
            stockid = db.execute("SELECT stockid from stocks WHERE ticker = :ticker", ticker = stocksold)[0]["stockid"]
            db.execute("INSERT INTO transactions (userid, quantity, price, stockid, trantype, time) VALUES (:userid, :qty, :price, :stockid, 'SELL', :time)",
            userid = session["user_id"], qty = -1*quantitysold, price = stockprice, stockid = stockid, time = time.strftime('%Y-%m-%d %H:%M:%S'))
            db.execute("UPDATE users SET cash = :newcash WHERE id = :userid", newcash = usercash + salevalue, userid = session["user_id"])
            return redirect("/")



def errorhandler(e):
    """Handle error"""
    return apology(e.name, e.code)


# listen for errors
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)
