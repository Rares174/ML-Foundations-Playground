#!/usr/bin/env python3
"""End-to-end workflow test"""

import urllib.request
import json
import time

print("=" * 70)
print("END-TO-END WORKFLOW TEST")
print("=" * 70)

# Test 1: Health check
print("\n✓ TEST 1: Health Check")
try:
    response = urllib.request.urlopen('http://127.0.0.1:5000/health')
    result = json.loads(response.read().decode())
    print(f"  ✅ /health: {result}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    exit(1)

# Test 2: Demo datasets accessible
print("\n✓ TEST 2: Demo Datasets")
for dataset in ['iris.csv', 'titanic.csv', 'house_prices.csv']:
    try:
        response = urllib.request.urlopen(f'http://127.0.0.1:5000/demo/{dataset}')
        lines = response.read().decode().split('\n')
        header = lines[0]
        rows = len([l for l in lines if l.strip()])
        print(f"  ✅ {dataset}: {rows} rows, columns: {len(header.split(','))}")
    except Exception as e:
        print(f"  ❌ {dataset} FAILED: {e}")
        exit(1)

# Test 3: Classification workflow (Iris + KNN)
print("\n✓ TEST 3: Classification Workflow (Iris + KNN)")
try:
    csv_file = r"c:\Users\usER\coding\ProSoft@NT\ML-Foundations-Playground\backend\demo\iris.csv"
    with open(csv_file, 'rb') as f:
        csv_data = f.read()
    
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = []
    body.append(f'--{boundary}'.encode())
    body.append(b'Content-Disposition: form-data; name="file"; filename="iris.csv"')
    body.append(b'Content-Type: text/csv')
    body.append(b'')
    body.append(csv_data)
    body.append(f'--{boundary}'.encode())
    body.append(b'Content-Disposition: form-data; name="features"')
    body.append(b'')
    body.append(json.dumps(["sepal_length", "sepal_width", "petal_length", "petal_width"]).encode())
    body.append(f'--{boundary}'.encode())
    body.append(b'Content-Disposition: form-data; name="target"')
    body.append(b'')
    body.append(b'species')
    body.append(f'--{boundary}'.encode())
    body.append(b'Content-Disposition: form-data; name="algorithm"')
    body.append(b'')
    body.append(b'knn')
    body.append(f'--{boundary}--'.encode())
    
    body_bytes = b'\r\n'.join(body)
    request = urllib.request.Request('http://127.0.0.1:5000/train', data=body_bytes)
    request.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    
    response = urllib.request.urlopen(request)
    result = json.loads(response.read().decode())
    print(f"  ✅ KNN Classification: {result['metric_name']} = {result['metric_value']}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    exit(1)

# Test 4: Regression workflow (House Prices + Linear Regression)
print("\n✓ TEST 4: Regression Workflow (House Prices + Linear Regression)")
try:
    csv_file = r"c:\Users\usER\coding\ProSoft@NT\ML-Foundations-Playground\backend\demo\house_prices.csv"
    with open(csv_file, 'rb') as f:
        csv_data = f.read()
    
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
    features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "LSTAT"]
    body.append(json.dumps(features).encode())
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
    request = urllib.request.Request('http://127.0.0.1:5000/train', data=body_bytes)
    request.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    
    response = urllib.request.urlopen(request)
    result = json.loads(response.read().decode())
    print(f"  ✅ Linear Regression: {result['metric_name']} = {result['metric_value']}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    exit(1)

# Test 5: Multiple algorithms
print("\n✓ TEST 5: Different Algorithms (Iris)")
algorithms = ['logistic_regression', 'decision_tree', 'random_forest']
csv_file = r"c:\Users\usER\coding\ProSoft@NT\ML-Foundations-Playground\backend\demo\iris.csv"
with open(csv_file, 'rb') as f:
    csv_data = f.read()

for algo in algorithms:
    try:
        boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
        body = []
        body.append(f'--{boundary}'.encode())
        body.append(b'Content-Disposition: form-data; name="file"; filename="iris.csv"')
        body.append(b'Content-Type: text/csv')
        body.append(b'')
        body.append(csv_data)
        body.append(f'--{boundary}'.encode())
        body.append(b'Content-Disposition: form-data; name="features"')
        body.append(b'')
        body.append(json.dumps(["sepal_length", "sepal_width", "petal_length", "petal_width"]).encode())
        body.append(f'--{boundary}'.encode())
        body.append(b'Content-Disposition: form-data; name="target"')
        body.append(b'')
        body.append(b'species')
        body.append(f'--{boundary}'.encode())
        body.append(b'Content-Disposition: form-data; name="algorithm"')
        body.append(b'')
        body.append(algo.encode())
        body.append(f'--{boundary}--'.encode())
        
        body_bytes = b'\r\n'.join(body)
        request = urllib.request.Request('http://127.0.0.1:5000/train', data=body_bytes)
        request.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
        
        response = urllib.request.urlopen(request)
        result = json.loads(response.read().decode())
        print(f"  ✅ {algo}: {result['metric_name']} = {result['metric_value']}")
    except Exception as e:
        print(f"  ❌ {algo} FAILED: {e}")
        exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - COMPLETE WORKFLOW IS WORKING!")
print("=" * 70)
print("\nYou can now:")
print("  1. Open http://127.0.0.1:5000 in your browser")
print("  2. Click a demo dataset button (Iris, Titanic, or House Prices)")
print("  3. Select features and target column")
print("  4. Choose an algorithm")
print("  5. Click 'Train Model' button")
print("  6. See the metric result displayed")
