# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS # Para permitir requisições do frontend
from genetic_algorithm import run_genetic_algorithm, load_data_once

app = Flask(__name__)
CORS(app) # Habilita CORS para todas as rotas

# Carrega os dados apenas uma vez quando o aplicativo Flask inicia
with app.app_context():
    load_data_once()

@app.route('/api/optimize', methods=['POST'])
def optimize_model():
    data = request.json
    pop_size = int(data.get('pop_size', 6))
    generations = int(data.get('generations', 10))
    mutation_rate = float(data.get('mutation_rate', 0.3))

    try:
        melhor_ind, acc, preds, labels, historico, tempo_total = run_genetic_algorithm(
            pop_size=pop_size,
            geracoes=generations,
            taxa_mutacao=mutation_rate
        )

        return jsonify({
            "status": "success",
            "message": "Otimização concluída com sucesso!",
            "results": {
                "best_individual": melhor_ind,
                "final_accuracy": acc,
                "total_time_seconds": tempo_total,
                "history_accuracies": historico,
                "predictions": preds, # Retorna as predições e labels para plotagem no frontend
                "true_labels": labels
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({"status": "ready", "message": "Backend is running and data is loaded."})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)