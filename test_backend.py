import requests
import os

BACKEND = "http://127.0.0.1:5000"

def test_demo_dataset(name):
    url = f"{BACKEND}/demo-dataset/{name}"
    r = requests.get(url)
    print(f"GET /demo-dataset/{name}: {r.status_code}")
    if r.status_code == 200:
        print(f"  ✓ {name} loaded, {len(r.text.splitlines())} lines")
    else:
        print(f"  ✗ {r.json().get('error')}")
    return r

def test_supervised():
    print("\nTesting supervised model (Titanic, logistic_regression)...")
    url = f"{BACKEND}/train"
    files = {'file': open('backend/demo/titanic.csv', 'rb')}
    data = {
        'algorithm': 'logistic_regression',
        'features': '["Age","Fare","Pclass"]',
        'target': 'Survived'
    }
    r = requests.post(url, files=files, data=data)
    print(f"POST /train (supervised): {r.status_code}")
    print(r.json())
    return r

def test_kmeans():
    print("\nTesting KMeans clustering (Titanic)...")
    url = f"{BACKEND}/train"
    files = {'file': open('backend/demo/titanic.csv', 'rb')}
    data = {
        'algorithm': 'kmeans',
        'features': '["Age","Fare","Pclass"]',
        'target': ''
    }
    r = requests.post(url, files=files, data=data)
    print(f"POST /train (kmeans): {r.status_code}")
    print(r.json())
    return r

if __name__ == "__main__":
    print("Testing backend demo datasets:")
    for name in ["iris", "titanic", "housing"]:
        test_demo_dataset(name)
    test_supervised()
    test_kmeans()
