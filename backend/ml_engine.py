import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import base64


def run_ml(file_path, algorithm, features, target):
    """
    ML pipeline supporting supervised and KMeans clustering, with matplotlib plots and explanations.
    """
    plots = []
    try:
        df = pd.read_csv(file_path)
        X = df[features].copy()
        # Drop rows with missing values in X
        mask = X.notna().all(axis=1)
        X = X[mask]
        if target:
            y = df[target].copy()[mask]
        else:
            y = None
        # Encode categorical columns in X
        X_encoded = X.copy()
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        # Encode target if categorical
        if y is not None:
            y_encoded = y.copy()
            if y_encoded.dtype == 'object':
                le_target = LabelEncoder()
                y_encoded = pd.Series(le_target.fit_transform(y_encoded.astype(str)), index=y_encoded.index)
        else:
            y_encoded = None

        # Supervised learning
        supervised_algs = ["linear_regression", "logistic_regression", "knn", "decision_tree", "random_forest"]
        if algorithm in supervised_algs:
            if y_encoded is None:
                raise ValueError("Target column required for supervised learning.")
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y_encoded, test_size=0.2, random_state=42
            )
            # Choose model and metric
            if algorithm == "linear_regression":
                model = LinearRegression()
                metric_name = "MAE"
                explanation = "MAE (Mean Absolute Error) measures the average magnitude of errors in predictions."
            elif algorithm == "logistic_regression":
                model = LogisticRegression(max_iter=1000, random_state=42)
                metric_name = "Accuracy"
                explanation = "Accuracy represents the percentage of correct predictions."
            elif algorithm == "knn":
                model = KNeighborsClassifier(n_neighbors=5)
                metric_name = "Accuracy"
                explanation = "Accuracy represents the percentage of correct predictions."
            elif algorithm == "decision_tree":
                if len(y_encoded.unique()) > (len(y_encoded) * 0.3):
                    model = LinearRegression()
                    metric_name = "MAE"
                    explanation = "MAE (Mean Absolute Error) measures the average magnitude of errors in predictions."
                else:
                    model = DecisionTreeClassifier(random_state=42)
                    metric_name = "Accuracy"
                    explanation = "Accuracy represents the percentage of correct predictions."
            elif algorithm == "random_forest":
                if len(y_encoded.unique()) > (len(y_encoded) * 0.3):
                    model = RandomForestRegressor(n_estimators=10, random_state=42)
                    metric_name = "MAE"
                    explanation = "MAE (Mean Absolute Error) measures the average magnitude of errors in predictions."
                else:
                    model = RandomForestClassifier(n_estimators=10, random_state=42)
                    metric_name = "Accuracy"
                    explanation = "Accuracy represents the percentage of correct predictions."
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            # Train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # Compute metric
            if metric_name == "Accuracy":
                metric_value = accuracy_score(y_test, y_pred)
            else:
                metric_value = mean_absolute_error(y_test, y_pred)
            metric_value = round(float(metric_value), 4)
            plot_dir = os.path.join(os.path.dirname(file_path), "ml_plots")
            os.makedirs(plot_dir, exist_ok=True)
            # Plotting logic by model
            if algorithm == "linear_regression":
                # For regression, plot predicted vs actual scatter
                plt.figure()
                plt.scatter(y_test, y_pred, alpha=0.7)
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title("Predicted vs Actual (Regression)")
                plot_path = os.path.join(plot_dir, f"{os.path.basename(file_path)}_regression_pred_vs_actual.png")
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                plots.append({
                    "path": plot_path,
                    "explanation": "Scatter plot of predicted vs actual values. Points close to the diagonal indicate good predictions."
                })
                # Also plot each feature vs target
                for feat in features:
                    plt.figure()
                    plt.scatter(df[feat], df[target], alpha=0.7)
                    plt.xlabel(feat)
                    plt.ylabel(target)
                    plt.title(f"{feat} vs {target}")
                    plot_path = os.path.join(plot_dir, f"{os.path.basename(file_path)}_{feat}_vs_{target}.png")
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    plt.close()
                    plots.append({
                        "path": plot_path,
                        "explanation": f"Scatter plot of {feat} vs {target}. Shows the relationship between this feature and the target."
                    })
            elif algorithm == "logistic_regression":
                # For classification, plot confusion matrix
                from sklearn.metrics import ConfusionMatrixDisplay
                plt.figure()
                ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
                plt.title("Confusion Matrix (Logistic Regression)")
                plot_path = os.path.join(plot_dir, f"{os.path.basename(file_path)}_logreg_confusion_matrix.png")
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                plots.append({
                    "path": plot_path,
                    "explanation": "Confusion matrix: shows how many samples were correctly or incorrectly classified."
                })
            elif algorithm == "knn":
                # For KNN, plot confusion matrix
                from sklearn.metrics import ConfusionMatrixDisplay
                plt.figure()
                ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
                plt.title("Confusion Matrix (KNN)")
                plot_path = os.path.join(plot_dir, f"{os.path.basename(file_path)}_knn_confusion_matrix.png")
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                plots.append({
                    "path": plot_path,
                    "explanation": "Confusion matrix: shows how many samples were correctly or incorrectly classified by KNN."
                })
            elif algorithm == "decision_tree" or algorithm == "random_forest":
                # For tree/forest, plot feature importances if available
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    plt.figure()
                    plt.barh(features, importances)
                    plt.xlabel("Importance")
                    plt.title("Feature Importances")
                    plot_path = os.path.join(plot_dir, f"{os.path.basename(file_path)}_{algorithm}_feature_importances.png")
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    plt.close()
                    plots.append({
                        "path": plot_path,
                        "explanation": "Feature importances: higher bars mean the feature is more important for prediction."
                    })
                # Also plot confusion matrix for classification
                if metric_name == "Accuracy":
                    from sklearn.metrics import ConfusionMatrixDisplay
                    plt.figure()
                    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
                    plt.title(f"Confusion Matrix ({algorithm.replace('_', ' ').title()})")
                    plot_path = os.path.join(plot_dir, f"{os.path.basename(file_path)}_{algorithm}_confusion_matrix.png")
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    plt.close()
                    plots.append({
                        "path": plot_path,
                        "explanation": "Confusion matrix: shows how many samples were correctly or incorrectly classified."
                    })
                # For regression, plot predicted vs actual
                if metric_name == "MAE":
                    plt.figure()
                    plt.scatter(y_test, y_pred, alpha=0.7)
                    plt.xlabel("Actual")
                    plt.ylabel("Predicted")
                    plt.title(f"Predicted vs Actual ({algorithm.replace('_', ' ').title()})")
                    plot_path = os.path.join(plot_dir, f"{os.path.basename(file_path)}_{algorithm}_regression_pred_vs_actual.png")
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    plt.close()
                    plots.append({
                        "path": plot_path,
                        "explanation": "Scatter plot of predicted vs actual values. Points close to the diagonal indicate good predictions."
                    })
        # KMeans clustering
        elif algorithm == "kmeans":
            # Only use features, ignore target
            if X_encoded.shape[1] < 2:
                raise ValueError("KMeans requires at least 2 features.")
            k = 3
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_encoded)
            metric_name = "inertia"
            metric_value = round(float(kmeans.inertia_), 4)
            explanation = "Inertia measures how internally coherent clusters are (lower is better)."
            # Plot: first two features, color by cluster
            plt.figure()
            plt.scatter(X_encoded.iloc[:, 0], X_encoded.iloc[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
            plt.xlabel(features[0])
            plt.ylabel(features[1])
            plt.title(f"KMeans Clusters ({features[0]} vs {features[1]})")
            plot_dir = os.path.join(os.path.dirname(file_path), "ml_plots")
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"{os.path.basename(file_path)}_kmeans.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            plots.append({
                "path": plot_path,
                "explanation": f"Scatter plot of the first two features colored by cluster assignment. Each color is a different cluster."
            })
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

        # Return metric, explanation, and plot info (list of dicts)
        return {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "explanation": explanation,
            "plots": plots
        }
    except Exception as e:
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)
        raise e
