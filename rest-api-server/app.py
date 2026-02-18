import os
import re
from flask import Flask, Response, request, jsonify, make_response, abort
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.json_util import dumps
from bson.objectid import ObjectId
from bson.errors import InvalidId

load_dotenv()

app = Flask(__name__)
mongo_db_url = os.environ.get("MONGO_DB_CONN_STRING")

client = MongoClient(mongo_db_url)
db = client['AAPLdb']

ALLOWED_STOCK_FIELDS = {"Date", "High", "Low", "Open", "Close", "Volume", "compound"}


def validate_object_id(id_str):
    try:
        return ObjectId(id_str)
    except (InvalidId, TypeError):
        abort(400, description="Invalid ObjectId format")


def validate_stock_data(data):
    if not data or not isinstance(data, dict):
        abort(400, description="Request body must be a non-empty JSON object")
    unknown = set(data.keys()) - ALLOWED_STOCK_FIELDS
    if unknown:
        abort(400, description=f"Unknown fields: {', '.join(unknown)}")
    if "Date" in data and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(data["Date"])):
        abort(400, description="Date must be in YYYY-MM-DD format")
    for field in ("High", "Low", "Open", "Close", "Volume"):
        if field in data and not isinstance(data[field], (int, float)):
            abort(400, description=f"{field} must be a number")


#Create
@app.post("/api/StockData")
def add_StockData():
    _json = request.json
    validate_stock_data(_json)
    db.StockData.insert_one(_json)

    resp = jsonify({"message": "StockData added successfully"})
    resp.status_code = 201
    return resp

#Read
@app.get("/api/StockData")
def get_StockData():
    Date = request.args.get('Date')
    if Date and not re.match(r"^\d{4}-\d{2}-\d{2}$", Date):
        abort(400, description="Date must be in YYYY-MM-DD format")
    query_filter = {} if Date is None else {"Date": Date}
    StockData = list(db.StockData.find(query_filter))

    response = Response(
        response=dumps(StockData), status=200,  mimetype="application/json")
    return response

#Update
@app.put("/api/StockData/<id>")
def update_StockData(id):
    oid = validate_object_id(id)
    _json = request.json
    validate_stock_data(_json)
    db.StockData.update_one({'_id': oid}, {"$set": _json})

    resp = jsonify({"message": "StockData updated successfully"})
    resp.status_code = 200
    return resp

#Delete
@app.delete("/api/StockData/<id>")
def delete_StockData(id):
    oid = validate_object_id(id)
    db.StockData.delete_one({'_id': oid})

    resp = jsonify({"message": "StockData deleted successfully"})
    resp.status_code = 200
    return resp


@app.errorhandler(400)
def handle_400_error(error):
    return make_response(jsonify({"errorCode": error.code, 
                                  "errorDescription": "Bad request!",
                                  "errorDetailedDescription": error.description,
                                  "errorName": error.name}), 400)

@app.errorhandler(404)
def handle_404_error(error):
        return make_response(jsonify({"errorCode": error.code, 
                                  "errorDescription": "Resource not found!",
                                  "errorDetailedDescription": error.description,
                                  "errorName": error.name}), 404)

@app.errorhandler(500)
def handle_500_error(error):
        return make_response(jsonify({"errorCode": error.code, 
                                  "errorDescription": "Internal Server Error",
                                  "errorDetailedDescription": error.description,
                                  "errorName": error.name}), 500)


if __name__ == '__main__':
     app.run()