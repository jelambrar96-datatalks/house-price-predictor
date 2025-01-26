import os
import requests

FLASK_SERVER = os.getenv("FLASK_SERVER", None); 

if not FLASK_SERVER is None:
    url = f"{FLASK_SERVER}/reload"
    response = requests.post(url=url)
    print(response.json())
else:
    print("FLASK_SERVER not defined")
