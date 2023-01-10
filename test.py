import requests

url = "https://txa53tjffl.execute-api.eu-west-3.amazonaws.com/test/predict"
data = {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/39/Typical_Street_In_The_Royal_Borough_Of_Kensington_And_Chelsea_In_London.jpg/375px-Typical_Street_In_The_Royal_Borough_Of_Kensington_And_Chelsea_In_London.jpg"}
result = requests.post(url, json=data).json()

print(result)