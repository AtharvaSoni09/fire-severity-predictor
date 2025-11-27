document.getElementById('prediction-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const btn = document.getElementById('predict-btn');
    const originalText = btn.innerText;
    btn.innerText = 'Analyzing...';
    btn.disabled = true;

    // Gather form data
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.success) {
            // Show results section
            const resultsSection = document.getElementById('results-section');
            resultsSection.style.display = 'flex';

            // Update Acres
            document.getElementById('acres-value').innerText = result.prediction_acres.toFixed(2);

            // Update Graph
            document.getElementById('lime-plot-img').src = 'data:image/png;base64,' + result.lime_plot;

            // Update Factors List
            const limeList = document.getElementById('lime-list');
            limeList.innerHTML = ''; // Clear previous

            // result.lime_data is list of [feature_condition, weight]
            // Sort by absolute weight to show most important first
            result.lime_data.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

            result.lime_data.forEach(item => {
                const feature = item[0];
                const weight = item[1];

                const li = document.createElement('li');
                li.className = 'lime-item';

                // Determine color/class based on weight sign
                // Weight > 0 means it contributes positively to the prediction value (Higher Fire Size)
                // Weight < 0 means it lowers the prediction
                const isIncrease = weight > 0;
                const color = isIncrease ? '#e84118' : '#4cd137'; // Red for increase, Green for decrease
                const direction = isIncrease ? 'Increases Risk' : 'Decreases Risk';

                li.style.borderLeft = `4px solid ${color}`;

                li.innerHTML = `
                    <div>
                        <div class="factor-name">${feature}</div>
                        <div style="font-size: 0.8rem; color: ${color};">${direction}</div>
                    </div>
                    <div class="factor-val">${weight.toFixed(4)}</div>
                `;

                limeList.appendChild(li);
            });

            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });

        } else {
            alert('Error: ' + result.error);
        }

    } catch (error) {
        console.error('Error:', error);
        const debugDiv = document.createElement('div');
        debugDiv.style.color = 'red';
        debugDiv.style.padding = '1rem';
        debugDiv.innerText = 'JS Error: ' + error.message;
        document.body.appendChild(debugDiv);
        alert('An error occurred: ' + error.message);
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
});
