#!/usr/bin/env python3
"""Test script for house prices regression"""

import urllib.request
import json
import time

# Wait a moment for Flask to fully start
time.sleep(1)

# Read the house_prices.csv file
csv_file_path = r"c:\Users\usER\coding\ProSoft@NT\ML-Foundations-Playground\backend\demo\house_prices.csv"
with open(csv_file_path, 'rb') as f:
    csv_data = f.read()

# Prepare the multipart form data
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = []
body.append(f'--{boundary}'.encode())
body.append(b'Content-Disposition: form-data; name="file"; filename="house_prices.csv"')
body.append(b'Content-Type: text/csv')
body.append(b'')
body.append(csv_data)
body.append(f'--{boundary}'.encode())
body.append(b'Content-Disposition: form-data; name="features"')
body.append(b'')
body.append(json.dumps(["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "LSTAT"]).encode())
body.append(f'--{boundary}'.encode())
body.append(b'Content-Disposition: form-data; name="target"')
body.append(b'')
body.append(b'MEDV')
body.append(f'--{boundary}'.encode())
body.append(b'Content-Disposition: form-data; name="algorithm"')
body.append(b'')
body.append(b'linear_regression')
body.append(f'--{boundary}--'.encode())

body_bytes = b'\r\n'.join(body)

# Send the request
url = 'http://127.0.0.1:5000/train'
request = urllib.request.Request(url, data=body_bytes)
request.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')

try:
    response = urllib.request.urlopen(request)
    result = json.loads(response.read().decode())
    print("✅ REGRESSION TRAINING SUCCESSFUL")
    print(f"Result: {result}")
    
    # Validate result format
    if "metric" in result and "value" in result:
        print(f"\n✅ Result format is correct:")
        print(f"   Metric: {result['metric']}")
        print(f"   Value: {result['value']}")
    else:
        print(f"\n❌ Result format incorrect")
        print(f"   Got: {result}")
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
