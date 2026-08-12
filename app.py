import os
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd

app = Flask(__name__)
app.secret_key = 'placement_prediction_secret'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/dataset", methods=["GET", "POST"])
def dataset():
    table_html = None
    summary_html = None
    stats = {"total_students": None, "total_features": None,
             "missing_values": None, "duplicate_records": None}
    error = None

    filepath = session.get('dataset_path')
    if filepath and os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            stats["total_students"] = len(df)
            stats["total_features"] = len(df.columns)
            stats["missing_values"] = int(df.isnull().sum().sum())
            stats["duplicate_records"] = int(df.duplicated().sum())
        except Exception as e:
            error = str(e)

    if request.method == "POST":
        action = request.form.get("action")

        if action == "upload":
            file = request.files.get("dataset_file")
            if file and file.filename.endswith(".csv"):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                session['dataset_path'] = filepath
                return redirect(url_for('dataset'))
            else:
                error = "Please upload a valid CSV file."

        elif action == "view":
            if filepath and os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    table_html = df.head(50).to_html(
                        classes="dataset-table", index=False, border=0)
                except Exception as e:
                    error = str(e)
            else:
                error = "No dataset uploaded yet."

        elif action == "summary":
            if filepath and os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    summary_html = df.describe(include='all').to_html(
                        classes="dataset-table", border=0)
                except Exception as e:
                    error = str(e)
            else:
                error = "No dataset uploaded yet."

    return render_template("dataset.html", stats=stats,
                           table_html=table_html,
                           summary_html=summary_html,
                           error=error)


@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html")


@app.route("/visualization")
def visualization():
    return render_template("visualization.html")


@app.route("/models")
def models():
    return render_template("models.html")


@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/reports")
def reports():
    return render_template("reports.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)