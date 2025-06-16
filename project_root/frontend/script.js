// script.js
document.getElementById('aiSettingsForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const submitBtn = document.querySelector('.submit-btn');
    submitBtn.textContent = 'Otimizando...';
    submitBtn.disabled = true;

    const popSize = document.getElementById('popSize').value;
    const generations = document.getElementById('generations').value;
    const mutationRate = document.getElementById('mutationRate').value / 100; // Converte para float

    const data = {
        pop_size: parseInt(popSize),
        generations: parseInt(generations),
        mutation_rate: parseFloat(mutationRate)
    };

    const resultsDiv = document.getElementById('results');
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = '';
    resultsDiv.style.display = 'none';

    try {
        const response = await fetch('http://localhost:5000/api/optimize', { // Mude para o IP do seu backend se não for localhost
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Erro HTTP: ${response.status} - ${errorText}`);
        }

        const result = await response.json();
        console.log('Resultados da Otimização:', result);

        if (result.status === "success") {
            document.getElementById('status').textContent = 'Concluído';
            document.getElementById('bestIndividual').textContent = JSON.stringify(result.results.best_individual, null, 2);
            document.getElementById('finalAccuracy').textContent = result.results.final_accuracy.toFixed(4);
            document.getElementById('totalTime').textContent = `${result.results.total_time_seconds.toFixed(1)} s`;

            const historyList = document.getElementById('historyList');
            historyList.innerHTML = '';
            result.results.history_accuracies.forEach((genAccs, index) => {
                const li = document.createElement('li');
                li.textContent = `Geração ${index + 1}: ${genAccs.map(a => a.toFixed(4)).join(', ')}`;
                historyList.appendChild(li);
            });

            // Para exibir imagens, o backend precisaria retornar as imagens ou IDs
            // No momento, seu backend retorna preds e labels, o que é útil para análise mas não para exibir a imagem real no frontend.
            // Para exibir imagens, você precisaria de um endpoint no backend que retorne imagens
            // dado um índice ou uma forma de pré-renderizar e servir as imagens.
            // Por simplicidade, vou pular a parte de plotagem de imagem no front-end por agora,
            // mas você pode estender isso criando um endpoint no backend que serve imagens.
            resultsDiv.style.display = 'block';

        } else {
            errorMessage.textContent = `Erro na otimização: ${result.message}`;
        }

    } catch (error) {
        console.error('Erro ao conectar ou processar:', error);
        errorMessage.textContent = `Falha ao otimizar: ${error.message}`;
    } finally {
        submitBtn.textContent = 'Iniciar Otimização';
        submitBtn.disabled = false;
    }
});