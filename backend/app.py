from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from ml_engine import run_ml
import os
import tempfile
import json

frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")
app = Flask(__name__, static_folder=frontend_path, static_url_path="")
CORS(app)
from flask import abort

# Legacy endpoint for /demo/<filename> for test compatibility
@app.route('/demo/<filename>', methods=['GET'])
def serve_demo_legacy(filename):
    file_path = os.path.join(DEMO_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": f"Demo file '{filename}' not found"}), 404
    return send_file(file_path, mimetype='text/csv', as_attachment=False)
# Serve matplotlib plot images by path (for frontend display)
@app.route('/ml-plot')
def serve_ml_plot():
    plot_path = request.args.get('path', '')
    # Only allow files in temp or backend/demo/ml_plots for safety
    allowed_dirs = [tempfile.gettempdir(), os.path.join(os.path.dirname(__file__), 'ml_plots')]
    plot_path = os.path.abspath(plot_path)
    if not any(plot_path.startswith(os.path.abspath(d)) for d in allowed_dirs):
        return abort(403)
    if not os.path.exists(plot_path):
        return abort(404)
    return send_file(plot_path, mimetype='image/png', as_attachment=False)

# Create temp directory for uploads
UPLOAD_FOLDER = tempfile.gettempdir()
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Demo datasets directory
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo")


@app.route('/train', methods=['POST'])
def train_model():
    """
    Train ML model endpoint (supports supervised and kmeans clustering).
    """
    try:
        # Check for file
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Parse algorithm (singular)
        algorithm = request.form.get('algorithm', '').strip().lower()
        if not algorithm:
            return jsonify({"error": "No algorithm specified"}), 400

        # Parse features
        features_str = request.form.get('features', '')
        if features_str.startswith('['):
            try:
                features = json.loads(features_str)
            except:
                features = [f.strip() for f in features_str.split(',') if f.strip()]
        else:
            features = [f.strip() for f in features_str.split(',') if f.strip()]
        if not features:
            return jsonify({"error": "No features specified"}), 400

        # Parse target (may be ignored for kmeans)
        target = request.form.get('target', '').strip()


        # Parse metric (optional)
        metric = request.form.get('metric', '').strip().lower() or None

        # Dynamic pipeline logic
        supervised_algs = [
            "linear_regression", "logistic_regression", "knn", "decision_tree", "random_forest",
            "random_forest_classifier", "support_vector_machine", "naive_bayes", "gradient_boosting", "xgboost"
        ]
        if algorithm in supervised_algs:
            if not target:
                return jsonify({"error": "Target column required for supervised learning."}), 400
        elif algorithm == "kmeans":
            # Ignore target for clustering
            target = None
        else:
            return jsonify({"error": f"Unknown algorithm: {algorithm}"}), 400

        # Save file temporarily
        temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(temp_path)

        # Run ML pipeline (returns dict with metric, explanation, plots)
        try:
            result = run_ml(temp_path, algorithm, features, target, metric=metric)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        return jsonify(result), 200

    except Exception as e:
        print(f"ERROR in /train: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200



# New endpoint: /demo-dataset/<name>
@app.route('/demo-dataset/<name>', methods=['GET'])
def serve_demo_dataset(name):
    """Serve demo CSV by logical name (iris, titanic, housing)."""
    name_map = {
        "iris": "iris.csv",
        "titanic": "titanic.csv",
        "housing": "house_prices.csv"
    }
    fname = name_map.get(name.lower())
    if not fname:
        return jsonify({"error": f"Unknown demo dataset: {name}"}), 404
    file_path = os.path.join(DEMO_DIR, fname)
    if not os.path.exists(file_path):
        return jsonify({"error": f"Demo file '{fname}' not found"}), 404
    return send_file(file_path, mimetype='text/csv', as_attachment=False)


@app.route('/')
def index():
    """Serve frontend."""
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
