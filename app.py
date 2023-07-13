from flask import Flask
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    # Run your specified Python script
    script_path = './testrunai.py'
    result = subprocess.run(['python', script_path], capture_output=True, text=True)

    # Print subprocess output for debugging
    print(result.stdout)

    # Display the output in the browser
    return f"<pre>{result.stdout}</pre>"
@app.route('/predict', methods=['POST'])
def predict():
    return "Hello World!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug= True)

