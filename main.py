import os
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load('artefatos/iris_model.joblib')

class_names = ['setosa', 'versicolor', 'virginica']

SECRET_KEY = 'sua_chave_secreta_aqui'

# Função para verificar a chave de API
def check_api_key(req):
    # se for conectar no banco, precisa buscar a chave na tabela
    # e comparar com a chave que está no header
    
    # Aqui estamos apenas comparando com uma chave secreta fixa
    # mas você pode implementar uma lógica de autenticação mais robusta
    key = req.headers.get('x-api-key')

    return key == SECRET_KEY

@app.route('/predict', methods=['POST'])
def predict():

    if not check_api_key(request):
        return jsonify({'error': 'Acesso não autorizado, chave inválida'}), 401
    
    data = request.get_json()

    if not data or 'features' not in data:
        return jsonify({'error': 'Dados inválidos, verifique o JSON enviado'}), 500
    
    features = data['features']

    if len(features) != 4:
        return jsonify({'error': 'Número de características inválido, deve ser 4'}), 500

    try:
        prediction = model.predict([features])



        return jsonify({'prediction': class_names[prediction[0]]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint para verificar a saúde da aplicação
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host="0.0.0.0", port=port)