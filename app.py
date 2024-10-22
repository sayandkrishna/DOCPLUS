from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

# Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Alzheimer's Disease detection page
@app.route('/alzheimers')
def alzheimers():
    # Redirect to the Streamlit app for Alzheimer's detection
    streamlit_url = "http://localhost:8501"  # Change this to your production URL when deployed
    try:
        return redirect(streamlit_url)
    except Exception as e:
        print("Error redirecting to Streamlit app:", e)
        return render_template('index.html', error="Could not redirect to the Alzheimer's detection app.")

# Chest scan detection page
@app.route('/chest-scan')
def chest_scan():
    # Redirect to the Streamlit app for chest scan detection
    streamlit_url = "http://localhost:8502"  # Change this to your production URL when deployed
    try:
        return redirect(streamlit_url)
    except Exception as e:
        print("Error redirecting to Streamlit app:", e)
        return render_template('index.html', error="Could not redirect to the chest scan detection app.")

if __name__ == '__main__':
    app.run(debug=True)
