// Global variables
const API_BASE = '/api/v1';
let currentTab = 'prediction';  // Start on prediction

// Tab navigation
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.style.display = 'none';
    });
    
    // Show selected tab
    const selectedTab = document.getElementById(tabName);
    if (selectedTab) {
        selectedTab.style.display = 'block';
    }
    
    // Update nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // Mark clicked nav item as active (find by onclick attribute)
    document.querySelectorAll('.nav-item').forEach(item => {
        if (item.getAttribute('onclick') && item.getAttribute('onclick').includes(`'${tabName}'`)) {
            item.classList.add('active');
        }
    });
    
    currentTab = tabName;
    loadTabData(tabName);
}

function showExplanationMethod(method) {
    document.querySelectorAll('.explanation-content').forEach(content => {
        content.style.display = 'none';
    });
    document.querySelectorAll('.explanation-tabs .tab').forEach(tab => {
        tab.classList.remove('active');
    });

   const methodDiv = document.getElementById(method + 'Explanation');
   if (methodDiv) {
       methodDiv.style.display = 'block';
   }
   event.target?.classList.add('active');
}

// Load tab-specific data
async function loadTabData(tabName) {
    switch(tabName) {
        case 'overview':
            loadOverviewData();
            break;
        case 'features':
            loadFeatureAnalysis();
            break;
        case 'prediction':
            setupPredictionForm();
            break;
        case 'comparison':
            // Disabled:
            break;
        case 'explanations':
            // Loaded after prediction
            break;
    }
}

// Overview (DISABLED)
async function loadOverviewData() {
    const statsDiv = document.getElementById('statsGrid');
    statsDiv.innerHTML = `
        <div class="stat-card primary">
            <div class="stat-value">-</div>
            <div class="stat-label">Total Predictions</div>
        </div>
        <div class="stat-card danger">
            <div class="stat-value">-</div>
            <div class="stat-label">High Risk Rate</div>
        </div>
        <div class="stat-card success">
            <div class="stat-value">-</div>
            <div class="stat-label">Avg Confidence</div>
        </div>
        <div class="stat-card warning">
            <div class="stat-value">-</div>
            <div class="stat-label">RF Usage</div>
        </div>
    `;
}

// Feature Analysis (DISABLED)
async function loadFeatureAnalysis() {
    const container = document.getElementById('features');
    container.innerHTML = `<p style="color: #999; padding: 20px;">Feature analysis dashboard coming soon...</p>`;
}

// Setup prediction form (WORKING)
function setupPredictionForm() {
    const form = document.getElementById('predictionForm');
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const eventData = {};
        
        for (let [key, value] of formData.entries()) {
            if (value === 'true' || value === 'false') {
                eventData[key] = value === 'true';
            } else if (!isNaN(value) && value !== '') {
                eventData[key] = parseFloat(value);
            } else {
                eventData[key] = value;
            }
        }
        
        await makePrediction(eventData);
    });
}

// Make prediction
async function makePrediction(eventData) {
    const resultsDiv = document.getElementById('predictionResults');
    resultsDiv.innerHTML = '<div class="loading">⏳ Making prediction...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(eventData)
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Display prediction results
        const riskColors = { 0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c' };
        const riskLabels = { 0: 'LOW', 1: 'MEDIUM', 2: 'HIGH' };
        
        resultsDiv.innerHTML = `
            <div style="margin-bottom: 20px;">
                <div class="risk-badge" style="background-color: ${riskColors[result.risk]}; color: white; padding: 12px 20px; border-radius: 8px; font-weight: bold; display: inline-block;">
                    ${riskLabels[result.risk]} RISK
                </div>
                <div style="margin-top: 16px;">
                    <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                    <p><strong>Model Used:</strong> ${result.model_used}</p>
                    <p><strong>Risk Class:</strong> ${result.risk_description}</p>
                </div>
            </div>
            
            <div style="margin-top: 20px; padding: 16px; background: #f8f9fa; border-radius: 8px;">
                <strong>Probabilities:</strong>
                <div style="margin-top: 12px;">
                    <p>🟢 Low Risk: ${(result.probabilities.normal * 100).toFixed(1)}%</p>
                    <p>🟡 Medium: ${(result.probabilities.medium * 100).toFixed(1)}%</p>
                    <p>🔴 High Risk: ${(result.probabilities.high_risk * 100).toFixed(1)}%</p>
                </div>
            </div>
        `;
        
        // Show explanation
        await displayExplanation(result);
        
        // Switch to explanations tab
        showTab('explanations');
        
    } catch (error) {
        resultsDiv.innerHTML = `<div class="error" style="color: #e74c3c; padding: 16px; background: #ffe6e6; border-radius: 8px;">❌ ${error.message}</div>`;
    }
}

// Display explanation (from /predict response)
async function displayExplanation(predictionResult) {
    const explanation = predictionResult.explanation;
    
    if (!explanation || !explanation.shap_analysis) {
        console.warn('No SHAP analysis in response');
        return;
    }
    
    const shap = explanation.shap_analysis;
    const container = document.getElementById('shapFactors');
    
    if (!container) return;
    
    // Extract local importance (array of objects)
    const factors = shap.local_importance || [];
    
    container.innerHTML = factors.map(factor => `
        <div class="feature-item" style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span><strong>${factor.feature}</strong></span>
                <span style="color: #e74c3c; font-weight: bold;">+${(factor.importance * 100).toFixed(1)}%</span>
            </div>
            <div class="shap-bar-container" style="height: 12px; background: #f0f0f0; border-radius: 4px; overflow: hidden;">
                <div class="shap-bar" style="width: ${Math.min(factor.importance * 100, 100)}%; height: 100%; background: linear-gradient(90deg, #f39c12, #e74c3c);"></div>
            </div>
            <div style="font-size: 11px; color: #999; margin-top: 4px;">Rank: ${factor.rank}</div>
        </div>
    `).join('');
    
    // Natural language explanation
    const whyFlaggedDiv = document.getElementById('whyFlagged');
    if (whyFlaggedDiv) {
        const briefReason = shap.brief_reason || 'Activity flagged by model.';
        whyFlaggedDiv.innerHTML = `
            <span style="color: #666;">📋 Why Flagged:</span>
            <p style="margin-top: 8px; color: #333; line-height: 1.6;">${briefReason}</p>
            <p style="margin-top: 12px; font-size: 12px; color: #999;">
                <strong>Mapping Confidence:</strong> ${(shap.mapping_confidence * 100).toFixed(1)}%
            </p>
        `;
    }
}

// Model comparison (DISABLED)
async function runModelComparison() {
    alert('Model comparison dashboard coming soon. Use Live Prediction to see individual model behavior.');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupPredictionForm();
});


// ===== BATCH UPLOAD & PREDICTIONS =====

function setupBatchUpload() {
    const uploadForm = document.getElementById('batchUploadForm');
    if (!uploadForm) return;
    
    uploadForm.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        try {
            const events = await parseFile(file);
            await runBatchPredictions(events);
        } catch (error) {
            alert(`❌ Upload error: ${error.message}`);
        }
    });
}

async function parseFile(file) {
    const text = await file.text();
    
    if (file.name.endsWith('.csv')) {
        return parseCSV(text);
    } else if (file.name.endsWith('.json')) {
        return JSON.parse(text);
    } else {
        throw new Error('Unsupported format. Use CSV or JSON.');
    }
}

function parseCSV(text) {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const events = [];
    
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => v.trim());
        const event = {};
        
        headers.forEach((header, idx) => {
            const value = values[idx];
            // Auto-detect types
            if (value === 'true' || value === 'false') {
                event[header] = value === 'true';
            } else if (!isNaN(value) && value !== '') {
                event[header] = parseFloat(value);
            } else {
                event[header] = value;
            }
        });
        
        events.push(event);
    }
    
    return events;
}

async function runBatchPredictions(events) {
    const resultsDiv = document.getElementById('batchResults');
    resultsDiv.innerHTML = `<div class="loading">⏳ Processing ${events.length} events...</div>`;
    
    try {
        const response = await fetch(`${API_BASE}/predict/batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(events)
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const result = await response.json();
        displayBatchResults(result);
        
    } catch (error) {
        resultsDiv.innerHTML = `<div class="error">❌ ${error.message}</div>`;
    }
}

function displayBatchResults(result) {
    const { results, metadata } = result;
    const resultsDiv = document.getElementById('batchResults');
    
    // Summary stats
    const stats = results.reduce((acc, r) => {
        if (r.error) acc.errors++;
        else {
            acc.total++;
            acc[`risk_${r.risk}`] = (acc[`risk_${r.risk}`] || 0) + 1;
            acc.avgConfidence += r.confidence;
        }
        return acc;
    }, { total: 0, errors: 0, risk_0: 0, risk_1: 0, risk_2: 0, avgConfidence: 0 });
    
    if (stats.total > 0) {
        stats.avgConfidence /= stats.total;
    }
    
    const riskColors = { 0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c' };
    const riskLabels = { 0: 'LOW', 1: 'MEDIUM', 2: 'HIGH' };
    
    resultsDiv.innerHTML = `
        <div style="margin-bottom: 20px; padding: 16px; background: #f8f9fa; border-radius: 8px;">
            <h4>📊 Batch Summary</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 12px;">
                <div style="padding: 12px; background: #667eea; color: white; border-radius: 6px;">
                    <div style="font-size: 24px; font-weight: bold;">${metadata.total_events}</div>
                    <div style="font-size: 12px;">Total Events</div>
                </div>
                <div style="padding: 12px; background: #2ecc71; color: white; border-radius: 6px;">
                    <div style="font-size: 24px; font-weight: bold;">${stats.risk_0}</div>
                    <div style="font-size: 12px;">🟢 Low Risk</div>
                </div>
                <div style="padding: 12px; background: #f39c12; color: white; border-radius: 6px;">
                    <div style="font-size: 24px; font-weight: bold;">${stats.risk_1}</div>
                    <div style="font-size: 12px;">🟡 Medium</div>
                </div>
                <div style="padding: 12px; background: #e74c3c; color: white; border-radius: 6px;">
                    <div style="font-size: 24px; font-weight: bold;">${stats.risk_2}</div>
                    <div style="font-size: 12px;">🔴 High Risk</div>
                </div>
                <div style="padding: 12px; background: #3498db; color: white; border-radius: 6px;">
                    <div style="font-size: 24px; font-weight: bold;">${(stats.avgConfidence * 100).toFixed(0)}%</div>
                    <div style="font-size: 12px;">Avg Confidence</div>
                </div>
                <div style="padding: 12px; background: #95a5a6; color: white; border-radius: 6px;">
                    <div style="font-size: 24px; font-weight: bold;">${stats.errors}</div>
                    <div style="font-size: 12px;">Errors</div>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 20px;">
            <h4>📋 Detailed Results</h4>
            <div style="max-height: 600px; overflow-y: auto;">
                ${results.map((r, idx) => {
                    if (r.error) {
                        return `
                            <div style="padding: 12px; margin-bottom: 12px; background: #ffe6e6; border-left: 4px solid #e74c3c; border-radius: 4px;">
                                <div style="font-weight: bold;">Event ${r.index}</div>
                                <div style="color: #e74c3c; font-size: 12px; margin-top: 4px;">❌ ${r.error}</div>
                            </div>
                        `;
                    }
                    
                    const shap = r.explanation?.shap_analysis || {};
                    const factors = shap.local_importance || [];
                    
                    return `
                        <div style="padding: 16px; margin-bottom: 16px; background: #f8f9fa; border: 1px solid #ddd; border-radius: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                <div>
                                    <strong>Event ${r.index}</strong>
                                    <span style="background: ${riskColors[r.risk]}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; margin-left: 12px;">
                                        ${riskLabels[r.risk]}
                                    </span>
                                </div>
                                <div style="font-size: 12px; color: #666;">
                                    Confidence: ${(r.confidence * 100).toFixed(1)}% | Model: ${r.model_used}
                                </div>
                            </div>
                            
                            <div style="margin: 12px 0;">
                                <p style="font-size: 12px; margin-bottom: 8px;"><strong>📊 Probabilities:</strong></p>
                                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 12px;">
                                    <div>🟢 Low: ${(r.probabilities.normal * 100).toFixed(1)}%</div>
                                    <div>🟡 Med: ${(r.probabilities.medium * 100).toFixed(1)}%</div>
                                    <div>🔴 High: ${(r.probabilities.high_risk * 100).toFixed(1)}%</div>
                                </div>
                            </div>
                            
                            ${factors.length > 0 ? `
                                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #ddd;">
                                    <p style="font-size: 12px; font-weight: bold; margin-bottom: 8px;">🔍 Top Features:</p>
                                    ${factors.slice(0, 3).map(f => `
                                        <div style="margin-bottom: 6px;">
                                            <div style="display: flex; justify-content: space-between; font-size: 11px;">
                                                <span>${f.feature}</span>
                                                <span style="color: #e74c3c; font-weight: bold;">+${(f.importance * 100).toFixed(1)}%</span>
                                            </div>
                                            <div style="height: 4px; background: #f0f0f0; border-radius: 2px; overflow: hidden;">
                                                <div style="width: ${Math.min(f.importance * 100, 100)}%; height: 100%; background: linear-gradient(90deg, #f39c12, #e74c3c);"></div>
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            ` : ''}
                            
                            ${shap.brief_reason ? `
                                <div style="margin-top: 12px; padding: 8px; background: white; border-radius: 4px; font-size: 11px; color: #333;">
                                    <strong>Why:</strong> ${shap.brief_reason}
                                </div>
                            ` : ''}
                        </div>
                    `;
                }).join('')}
            </div>
        </div>
        
        <div style="margin-top: 16px; font-size: 12px; color: #999;">
            ⏱️ Processing time: ${metadata.processing_time_ms.toFixed(0)}ms | Success rate: ${((metadata.successful_predictions / metadata.total_events) * 100).toFixed(1)}%
        </div>
    `;
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupPredictionForm();
    setupBatchUpload();
});