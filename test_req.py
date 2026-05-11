import urllib.request
import json
import urllib.error

url = 'http://localhost:8000/analyze'
data = json.dumps({'symptoms': 'I have a sore throat and fever'}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as res:
        print(res.read().decode())
except urllib.error.HTTPError as e:
    print(f"HTTPError: {e.code}")
    print(e.read().decode())
