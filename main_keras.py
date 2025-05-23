import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# Carrega o modelo UNet salvo
model = tf.keras.models.load_model('artefatos/unet_model.h5')

SECRET_KEY = 'sua_chave_secreta_aqui'

def check_api_key(req):
    key = req.headers.get('x-api-key')
    return key == SECRET_KEY

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((128, 128))  # ajuste para o tamanho esperado pelo modelo
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def postprocess_mask(mask):
    # máscara entre 0 e 1, limiar 0.5 para binarizar
    mask = (mask[0, :, :, 0] > 0.5).astype(np.uint8) * 255
    return mask.tolist()

@app.route('/predict', methods=['POST'])
def predict():
    if not check_api_key(request):
        return jsonify({'error': 'Acesso não autorizado, chave inválida'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'Arquivo de imagem não fornecido'}), 400
    
    file = request.files['image']
    image_bytes = file.read()

    try:
        input_img = preprocess_image(image_bytes)
        prediction = model.predict(input_img)
        mask = postprocess_mask(prediction)
        return jsonify({'mask': mask})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
