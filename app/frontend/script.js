const featureGrid = document.querySelector('#feature-grid');
const form = document.querySelector('#transaction-form');
const predictButton = document.querySelector('#predict-button');
const emptyResult = document.querySelector('#empty-result');
const predictionResult = document.querySelector('#prediction-result');
const errorResult = document.querySelector('#error-result');

for (let index = 1; index <= 28; index += 1) {
	const label = document.createElement('label');
	label.className = 'field feature';
	label.innerHTML = `V${index}<input name="V${index}" type="number" step="any" value="0" required>`;
	featureGrid.appendChild(label);
}

form.addEventListener('submit', async (event) => {
	event.preventDefault();
	predictButton.disabled = true;
	predictButton.innerHTML = 'Scoring... <span>↗</span>';
	errorResult.hidden = true;
	const transaction = Object.fromEntries(new FormData(form).entries());
	Object.keys(transaction).forEach((key) => { transaction[key] = Number(transaction[key]); });
	try {
		const response = await fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(transaction) });
		const result = await response.json();
		if (!response.ok) throw new Error(result.detail || 'The transaction could not be assessed.');
		showResult(result);
	} catch (error) {
		errorResult.textContent = error.message;
		errorResult.hidden = false;
		emptyResult.hidden = true;
		predictionResult.hidden = true;
	} finally {
		predictButton.disabled = false;
		predictButton.innerHTML = 'Check transaction <span>→</span>';
	}
});

document.querySelector('#reset-button').addEventListener('click', () => window.setTimeout(() => { emptyResult.hidden = false; predictionResult.hidden = true; errorResult.hidden = true; }, 0));

function showResult(result) {
	const probability = Number(result.fraud_probability) * 100;
	const isFraud = result.prediction === 'FRAUD';
	emptyResult.hidden = true;
	errorResult.hidden = true;
	predictionResult.hidden = false;
	document.querySelector('#prediction-value').textContent = result.prediction;
	document.querySelector('#prediction-value').style.color = isFraud ? '#ff9b8d' : '#d9ed72';
	document.querySelector('#probability-value').textContent = `${probability.toFixed(4)}%`;
	document.querySelector('#probability-meter').style.width = `${Math.min(probability, 100)}%`;
	document.querySelector('#threshold-value').textContent = `${Number(result.threshold) * 100}%`;
	document.querySelector('#result-time').textContent = new Date().toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' });
	document.querySelector('#result-message').textContent = isFraud ? 'This transaction crossed the configured risk threshold and needs review.' : 'This transaction is below the configured risk threshold.';
}
