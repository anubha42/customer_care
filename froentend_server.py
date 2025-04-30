from flask import Flask, render_template, request, jsonify
import requests
import os


app = Flask(__name__)

# Update this with the actual backend server URL
BACKEND_URL = "https://9k4a4bih5jork6-3500.proxy.runpod.net/process_audio"  # Replace with the backend server's IP and port



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Upload recorded audio to backend and fetch TTS output.
    """
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    # Send file to backend
    try:
        response = requests.post(BACKEND_URL, files={'file': file})
        if response.status_code == 200:
            # Save the file locally or provide a download URL
            output_path = "static/customer_reply.wav"
            with open(output_path, "wb") as f:
                f.write(response.content)
            return jsonify({"message": "File processed successfully", "download_url": f"/{output_path}"})
        else:
            return jsonify({"error": response.json().get("error", "Unknown error")}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Ensure the static folder exists
    os.makedirs("static", exist_ok=True)
    app.run(host='0.0.0.0', port=5001, debug=True)