import requests

data = {'data': [540, 1, 0, 162, 2.5, 1040, 671, 600]}  # Example input
response = requests.post('http://127.0.0.1:5000/predict', json=data)
print(response.json())