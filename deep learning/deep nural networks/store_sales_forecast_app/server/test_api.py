import requests

data = {
    "store_nbr": 1,
    "family": "GROCERY I",
    "days": 7
}
response = requests.post("http://127.0.0.1:5000/api/predict", json=data)
print(response.json())