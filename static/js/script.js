/**
 * AI News Detector - Two-Color Interactive Frontend Script
 * Strictly uses Dark Navy Blue (#14213D) and White (#FFFFFF)
 */

const SAMPLE_REAL_TEXT = "The Federal Reserve announced on Wednesday that it will hold its benchmark interest rate steady following a two-day policy meeting, citing sustained moderate economic growth and low unemployment across domestic labor markets while continuing to monitor global supply chain developments.";

const SAMPLE_FAKE_TEXT = "BREAKING BOMBSHELL: Whistleblowers have leaked classified footage exposing a secret underground bunker beneath Washington where deep state operatives have been genetically modifying reptilian aliens to manipulate presidential election ballots and control international weather patterns!";

document.addEventListener('DOMContentLoaded', () => {
    initTextCounter();
    initNewsForm();
    initClearButton();
});

/**
 * Text counter for textarea
 */
function initTextCounter() {
    const textarea = document.getElementById('newsInput');
    const counter = document.getElementById('charCounter');
    if (!textarea || !counter) return;

    function updateCounter() {
        const text = textarea.value.trim();
        const chars = text.length;
        const words = text ? text.split(/\s+/).length : 0;
        counter.textContent = `${chars.toLocaleString()} characters | ${words.toLocaleString()} words`;
    }

    textarea.addEventListener('input', updateCounter);
    updateCounter();
}

/**
 * Load Sample Real News
 */
function loadRealSample() {
    const textarea = document.getElementById('newsInput');
    if (!textarea) return;
    textarea.value = SAMPLE_REAL_TEXT;
    textarea.dispatchEvent(new Event('input'));
    textarea.focus();
}

/**
 * Load Sample Fake News
 */
function loadFakeSample() {
    const textarea = document.getElementById('newsInput');
    if (!textarea) return;
    textarea.value = SAMPLE_FAKE_TEXT;
    textarea.dispatchEvent(new Event('input'));
    textarea.focus();
}

/**
 * Clear textarea
 */
function initClearButton() {
    const clearBtn = document.getElementById('clearBtn');
    const textarea = document.getElementById('newsInput');
    if (!clearBtn || !textarea) return;

    clearBtn.addEventListener('click', () => {
        textarea.value = '';
        textarea.dispatchEvent(new Event('input'));
        textarea.focus();
    });
}

/**
 * Handle form AJAX submission with loading spinner and dynamic result rendering
 */
function initNewsForm() {
    const form = document.getElementById('newsForm');
    const textarea = document.getElementById('newsInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const analyzeSpinner = document.getElementById('analyzeSpinner');
    const analyzeIcon = document.getElementById('analyzeIcon');
    const analyzeBtnText = document.getElementById('analyzeBtnText');

    if (!form || !textarea) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const text = textarea.value.trim();
        if (!text) {
            alert('Please enter or paste news text to analyze.');
            textarea.focus();
            return;
        }

        // Set Loading State
        analyzeBtn.disabled = true;
        if (analyzeSpinner) analyzeSpinner.style.display = 'inline-block';
        if (analyzeIcon) analyzeIcon.style.display = 'none';
        if (analyzeBtnText) analyzeBtnText.textContent = 'Analyzing...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ news_text: text })
            });

            const data = await response.json();

            if (!response.ok || data.status === 'error') {
                throw new Error(data.error || 'Failed to analyze news text.');
            }

            renderPredictionResult(data);

        } catch (error) {
            console.error('Prediction Error:', error);
            alert(`Analysis Error: ${error.message}`);
        } finally {
            // Restore button state
            analyzeBtn.disabled = false;
            if (analyzeSpinner) analyzeSpinner.style.display = 'none';
            if (analyzeIcon) analyzeIcon.style.display = 'inline-block';
            if (analyzeBtnText) analyzeBtnText.textContent = 'Analyze News';
        }
    });
}

/**
 * Dynamically render prediction results using only Dark Navy and White
 */
function renderPredictionResult(res) {
    const resultSection = document.getElementById('resultSection');
    if (!resultSection) return;

    const outcome = res.final_prediction;

    let badgeClass = 'badge-uncertain';
    let badgeIcon = 'fa-circle-question';
    let badgeText = 'UNCERTAIN';

    if (outcome === 'REAL') {
        badgeClass = 'badge-real';
        badgeIcon = 'fa-check';
        badgeText = 'REAL NEWS';
    } else if (outcome === 'FAKE') {
        badgeClass = 'badge-fake';
        badgeIcon = 'fa-triangle-exclamation';
        badgeText = 'FAKE NEWS';
    }

    const lr = res.models.logistic_regression;
    const nb = res.models.naive_bayes;

    resultSection.innerHTML = `
        <div class="card result-card">
            <div class="result-header">
                <div class="result-badge-wrap">
                    <span class="result-badge ${badgeClass}">
                        <i class="fa-solid ${badgeIcon}"></i> ${badgeText}
                    </span>
                    <span class="voting-summary">
                        <i class="fa-solid fa-check-double"></i> ${res.vote_tally}
                    </span>
                </div>
                <div class="duration-badge">
                    <i class="fa-solid fa-stopwatch"></i> ${res.duration_ms} ms
                </div>
            </div>

            <!-- Model-wise Breakdown Cards (2 Models) -->
            <div class="models-grid">
                <!-- Logistic Regression -->
                <div class="model-item">
                    <div class="model-item-top">
                        <span class="model-name">Logistic Regression</span>
                        <span class="badge-mini ${lr.prediction === 'REAL' ? 'mini-real' : 'mini-fake'}">
                            ${lr.prediction}
                        </span>
                    </div>
                    <div class="confidence-bar-wrap">
                        <div class="confidence-labels">
                            <span>Confidence</span>
                            <strong>${lr.confidence}%</strong>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill" style="width: ${lr.confidence}%"></div>
                        </div>
                    </div>
                </div>

                <!-- Naive Bayes -->
                <div class="model-item">
                    <div class="model-item-top">
                        <span class="model-name">Multinomial Naive Bayes</span>
                        <span class="badge-mini ${nb.prediction === 'REAL' ? 'mini-real' : 'mini-fake'}">
                            ${nb.prediction}
                        </span>
                    </div>
                    <div class="confidence-bar-wrap">
                        <div class="confidence-labels">
                            <span>Confidence</span>
                            <strong>${nb.confidence}%</strong>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill" style="width: ${nb.confidence}%"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Disclaimer -->
            <div class="disclaimer-banner">
                <i class="fa-solid fa-circle-info"></i>
                <span><strong>Disclaimer:</strong> ${res.disclaimer}</span>
            </div>
        </div>
    `;

    resultSection.classList.add('visible');
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Initialize simple Dashboard Chart using ONLY Dark Navy Blue (#14213D) and White (#FFFFFF)
 */
function initTwoColorDashboardChart() {
    const dataElem = document.getElementById('dashboardMetricsJson');
    if (!dataElem) return;

    let metrics = {};
    try {
        metrics = JSON.parse(dataElem.textContent);
    } catch (e) {
        console.error('Failed to parse dashboard metrics JSON', e);
        return;
    }

    const perfCanvas = document.getElementById('modelPerformanceChart');
    if (perfCanvas && Object.keys(metrics).length > 0) {
        const modelNames = Object.keys(metrics);
        const accuracyData = modelNames.map(m => (metrics[m].accuracy * 100).toFixed(2));
        const precisionData = modelNames.map(m => (metrics[m].precision * 100).toFixed(2));
        const recallData = modelNames.map(m => (metrics[m].recall * 100).toFixed(2));
        const f1Data = modelNames.map(m => (metrics[m].f1_score * 100).toFixed(2));

        new Chart(perfCanvas, {
            type: 'bar',
            data: {
                labels: modelNames,
                datasets: [
                    {
                        label: 'Accuracy (%)',
                        data: accuracyData,
                        backgroundColor: '#14213D', // Solid Dark Navy
                        borderColor: '#14213D',
                        borderWidth: 1,
                        borderRadius: 3
                    },
                    {
                        label: 'Precision (%)',
                        data: precisionData,
                        backgroundColor: 'rgba(20, 33, 61, 0.75)', // Navy 75%
                        borderColor: '#14213D',
                        borderWidth: 1,
                        borderRadius: 3
                    },
                    {
                        label: 'Recall (%)',
                        data: recallData,
                        backgroundColor: 'rgba(20, 33, 61, 0.50)', // Navy 50%
                        borderColor: '#14213D',
                        borderWidth: 1,
                        borderRadius: 3
                    },
                    {
                        label: 'F1 Score (%)',
                        data: f1Data,
                        backgroundColor: 'rgba(20, 33, 61, 0.25)', // Navy 25%
                        borderColor: '#14213D',
                        borderWidth: 1,
                        borderRadius: 3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        min: 80,
                        max: 100,
                        grid: { color: 'rgba(20, 33, 61, 0.08)' },
                        ticks: {
                            callback: function(val) { return val + '%'; },
                            color: '#14213D',
                            font: { family: "'JetBrains Mono', monospace", size: 11 }
                        }
                    },
                    x: {
                        grid: { display: false },
                        ticks: {
                            color: '#14213D',
                            font: { family: "'Plus Jakarta Sans', sans-serif", weight: 700, size: 12 }
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            padding: 14,
                            color: '#14213D',
                            font: { family: "'Plus Jakarta Sans', sans-serif", size: 11, weight: 600 }
                        }
                    },
                    tooltip: {
                        backgroundColor: '#14213D',
                        titleColor: '#FFFFFF',
                        bodyColor: '#FFFFFF',
                        borderColor: '#14213D',
                        borderWidth: 1,
                        callbacks: {
                            label: function(ctx) { return ` ${ctx.dataset.label}: ${ctx.raw}%`; }
                        }
                    }
                }
            }
        });
    }
}
