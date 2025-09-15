from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from ocr_utils import extract_text

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/ocr', methods=['POST'])
def ocr_route():
    if 'receipt' not in request.files or 'amount' not in request.form:
        return jsonify({'error': 'Eksik alan'}), 400

    file = request.files['receipt']
    amount = request.form.get('amount')
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    text = extract_text(filepath)

    try:
        amt_float = float(amount)
    except ValueError:
        return jsonify({'error': 'Geçersiz tutar'}), 400

    if str(int(amt_float)) in text or f"{int(amt_float)} TL" in text:
        status = 'approved'
    else:
        status = 'pending'

    return jsonify({
        'status': status,
        'ocr_text': text
    })
