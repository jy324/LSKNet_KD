import base64, requests

url = 'http://servera100:9000/predict'
files = { 'image': open('demo/dota_demo.jpg', 'rb') }
data = { 'score_thr': 0.35, 'return_image': 'true' }
r = requests.post(url, files=files, data=data, timeout=60)
resp = r.json()
print('Latency(ms):', resp['time_ms'])
print('First det:', resp['detections'][:1])
if resp.get('image_base64'):
    with open('vis.png', 'wb') as f:
        f.write(base64.b64decode(resp['image_base64']))
