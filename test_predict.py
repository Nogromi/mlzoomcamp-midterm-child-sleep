import requests
from pandas import Timestamp
import pandas as pd

# url = 'http://localhost:9696/predict' # for local test
url = 'https://child-sleep-small.onrender.com/predict' # running instance for testing

### test 1
# 'awake': 1
print('test 1')
child={'series_id': {2380679: '18b61dd5aae8'},
 'step': {2380679: 495539},
 'timestamp': {2380679: '2018-01-20T08:44:55-0500'},
 'anglez': {2380679: -6.344099998474121},
 'enmo': {2380679: 0.21040000021457672}
 }
response = requests.post(url, json=child).json()
if response['awake'] == True:
    print(f"child is awake")
else:
    print(f"child sleeps")


### test 2
# 'awake': 0
print('test 2')
child2={'series_id': {2380188: '18b61dd5aae8'},
 'step': {2380188: 495048},
 'timestamp': {2380188: '2018-01-20T08:04:00-0500'},
 'anglez': {2380188: 34.02190017700195},
 'enmo': {2380188: 0.0}}


response = requests.post(url, json=child2).json()
if response['awake'] == True:
    print(f"child is awake")
else:
    print(f"child sleeps")