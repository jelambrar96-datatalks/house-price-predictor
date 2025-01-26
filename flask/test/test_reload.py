import os
import sys
import requests

FLASK_SERVER = "http://127.0.0.1:8000"
# FLASK_SERVER = "http://127.0.0.1:8001"

if FLASK_SERVER is None:
    print("FLASK_SERVER not defined")
    sys.exit()

url = f"{FLASK_SERVER}/api/reload"
try:
    response = requests.post(url=url)

    # Check if the response status code is 200
    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print(f"Failed with status code {response.status_code}: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
