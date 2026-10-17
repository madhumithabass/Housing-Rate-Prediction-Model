/**
 * Boston Housing AI (NeuroEstate v2.0)
 * Client-Side Controller: Reactive UI, Chart.js Visualizations, Dual-Sync Inputs, Batch CSV Processing
 */

(function () {
  'use strict';

  // State Management
  let appMetadata = {};
  let impactChart = null;
  let radarChart = null;
  let livePredictTimeout = null;
  let batchResultsData = [];
  let batchCurrentPage = 1;
  const batchPageSize = 10;
  let activeCategory = 'all';

  // DOM Elements Cache
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const livePredictToggle = document.getElementById('livePredictToggle');
  const predictBtn = document.getElementById('predictBtn');
  const predictSpinner = document.getElementById('predictSpinner');
  const randomizeBtn = document.getElementById('randomizeBtn');
  const resetDefaultsBtn = document.getElementById('resetDefaultsBtn');
  const toastContainer = document.getElementById('toastContainer');

  // Result Elements
  const priceDisplay = document.getElementById('priceDisplay');
  const adjustedPriceDisplay = document.getElementById('adjustedPriceDisplay');
  const tierBadge = document.getElementById('tierBadge');
  const confidenceRangeDisplay = document.getElementById('confidenceRangeDisplay');
  const percentileRankText = document.getElementById('percentileRankText');
  const percentileBar = document.getElementById('percentileBar');
  const percentileMarker = document.getElementById('percentileMarker');
  const baselineRefText = document.getElementById('baselineRefText');
  const lastEvaluatedTag = document.getElementById('lastEvaluatedTag');

  // Batch Elements
  const batchDropZone = document.getElementById('batchDropZone');
  const batchFileInput = document.getElementById('batchFileInput');
  const fileChosenBanner = document.getElementById('fileChosenBanner');
  const chosenFileName = document.getElementById('chosenFileName');
  const removeFileBtn = document.getElementById('removeFileBtn');
  const processBatchBtn = document.getElementById('processBatchBtn');
  const batchSpinner = document.getElementById('batchSpinner');
  const batchResultsSection = document.getElementById('batchResultsSection');
  const batchTotalCount = document.getElementById('batchTotalCount');
  const batchMeanVal = document.getElementById('batchMeanVal');
  const batchMinVal = document.getElementById('batchMinVal');
  const batchMaxVal = document.getElementById('batchMaxVal');
  const batchTableBody = document.getElementById('batchTableBody');
  const batchTableSearch = document.getElementById('batchTableSearch');
  const exportCsvBtn = document.getElementById('exportCsvBtn');
  const prevPageBtn = document.getElementById('prevPageBtn');
  const nextPageBtn = document.getElementById('nextPageBtn');
  const pageInfo = document.getElementById('pageInfo');

  let selectedBatchFile = null;

  // -------------------------------------------------------------
  // 1. Initialization
  // -------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', () => {
    loadMetadata();
    setupTheme();
    setupNavigation();
    setupInputSynchronization();
    setupCategoryFilter();
    setupPresetButtons();
    setupBatchUploader();
    setupCodeSnippets();
    
    // Initial prediction run with defaults
    executePrediction(true);
  });

  function loadMetadata() {
    try {
      const metaScript = document.getElementById('initialMetadata');
      if (metaScript && metaScript.textContent) {
        appMetadata = JSON.parse(metaScript.textContent);
      }
    } catch (e) {
      console.warn('Failed parsing embedded metadata, fetching from API:', e);
      fetch('/api/metadata')
        .then(res => res.json())
        .then(data => { appMetadata = data; })
        .catch(err => console.error('Error fetching metadata:', err));
    }
  }

  // -------------------------------------------------------------
  // 2. Theme Management (Dark / Light)
  // -------------------------------------------------------------
  function setupTheme() {
    const savedTheme = localStorage.getItem('neuroestate_theme') || 'dark';
    setTheme(savedTheme);

    themeToggleBtn.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      setTheme(newTheme);
    });
  }

  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('neuroestate_theme', theme);
    
    const icon = themeToggleBtn.querySelector('i');
    if (theme === 'dark') {
      icon.className = 'fa-solid fa-moon';
    } else {
      icon.className = 'fa-solid fa-sun';
    }

    // Refresh charts with updated theme colors if already initialized
    if (impactChart) impactChart.update();
    if (radarChart) radarChart.update();
  }

  function getThemeColors() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    return {
      textColor: isDark ? '#f8fafc' : '#0f172a',
      textMuted: isDark ? '#94a3b8' : '#64748b',
      gridColor: isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.08)',
      cardBg: isDark ? '#131b2e' : '#ffffff'
    };
  }

  // -------------------------------------------------------------
  // 3. Navigation Tabs
  // -------------------------------------------------------------
  function setupNavigation() {
    const navTabs = document.querySelectorAll('.nav-tab');
    navTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const targetTab = tab.getAttribute('data-tab');
        
        navTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        document.querySelectorAll('.tab-pane').forEach(pane => {
          pane.classList.remove('active');
        });

        const activePane = document.getElementById(`tab-${targetTab}`);
        if (activePane) {
          activePane.classList.add('active');
        }
      });
    });

    const infoModalBtn = document.getElementById('infoModalBtn');
    if (infoModalBtn) {
      infoModalBtn.addEventListener('click', () => {
        const archTab = document.querySelector('.nav-tab[data-tab="model-architecture"]');
        if (archTab) archTab.click();
      });
    }
  }

  // -------------------------------------------------------------
  // 4. Feature Category Filter
  // -------------------------------------------------------------
  function setupCategoryFilter() {
    const filterBtns = document.querySelectorAll('.cat-filter-btn');
    const featureCards = document.querySelectorAll('.feature-card');

    filterBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const cat = btn.getAttribute('data-category');
        activeCategory = cat;

        featureCards.forEach(card => {
          if (cat === 'all' || card.getAttribute('data-category') === cat) {
            card.style.display = 'block';
          } else {
            card.style.display = 'none';
          }
        });
      });
    });
  }

  // -------------------------------------------------------------
  // 5. Input Synchronization (Slider <-> Number Input)
  // -------------------------------------------------------------
  function setupInputSynchronization() {
    const featureIds = [
      'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 
      'RM', 'AGE', 'DIS', 'RAD', 'TAX', 
      'PTRATIO', 'B', 'LSTAT'
    ];

    featureIds.forEach(id => {
      const slider = document.getElementById(`range_${id}`);
      const numberInput = document.getElementById(`num_${id}`);

      if (!slider || !numberInput) return;

      // Slider -> Number
      slider.addEventListener('input', (e) => {
        numberInput.value = e.target.value;
        triggerReactiveValuation();
      });

      // Number -> Slider
      numberInput.addEventListener('input', (e) => {
        slider.value = e.target.value;
        triggerReactiveValuation();
      });
    });

    // Manual predict button click
    predictBtn.addEventListener('click', () => {
      executePrediction(false);
    });
  }

  function triggerReactiveValuation() {
    if (!livePredictToggle.checked) return;

    if (livePredictTimeout) clearTimeout(livePredictTimeout);
    livePredictTimeout = setTimeout(() => {
      executePrediction(false);
    }, 120);
  }

  function collectFormValues() {
    const featureIds = [
      'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 
      'RM', 'AGE', 'DIS', 'RAD', 'TAX', 
      'PTRATIO', 'B', 'LSTAT'
    ];
    const data = {};
    featureIds.forEach(id => {
      const numInput = document.getElementById(`num_${id}`);
      data[id] = numInput ? parseFloat(numInput.value) : 0.0;
    });
    return data;
  }

  function setFormValues(values) {
    Object.keys(values).forEach(id => {
      const slider = document.getElementById(`range_${id}`);
      const numberInput = document.getElementById(`num_${id}`);
      if (slider && numberInput) {
        slider.value = values[id];
        numberInput.value = values[id];
      }
    });
  }

  // -------------------------------------------------------------
  // 6. Presets & Controls
  // -------------------------------------------------------------
  function setupPresetButtons() {
    const presetButtons = document.querySelectorAll('.preset-btn[data-preset-id]');
    presetButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        presetButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const presetId = btn.getAttribute('data-preset-id');
        const presets = appMetadata.presets || [];
        const found = presets.find(p => p.id === presetId);
        if (found && found.values) {
          setFormValues(found.values);
          executePrediction(false);
          showToast(`Loaded preset: ${found.title}`, 'success');
        }
      });
    });

    // Randomize
    randomizeBtn.addEventListener('click', () => {
      presetButtons.forEach(b => b.classList.remove('active'));
      const features = appMetadata.features || [];
      const randomValues = {};

      features.forEach(f => {
        // Randomize within 25th - 75th percentile range + slight variance
        const min = f.q25 || f.min;
        const max = f.q75 || f.max;
        const raw = min + Math.random() * (max - min);
        randomValues[f.id] = f.step < 1 ? Math.round(raw * 100) / 100 : Math.round(raw);
      });

      setFormValues(randomValues);
      executePrediction(false);
      showToast('Generated random realistic property profile', 'success');
    });

    // Reset Defaults (Notebook Benchmark)
    resetDefaultsBtn.addEventListener('click', () => {
      presetButtons.forEach(b => b.classList.remove('active'));
      const benchmarkPreset = (appMetadata.presets || []).find(p => p.id === 'notebook_benchmark');
      if (benchmarkPreset) {
        setFormValues(benchmarkPreset.values);
      } else {
        const features = appMetadata.features || [];
        const defaults = {};
        features.forEach(f => { defaults[f.id] = f.default; });
        setFormValues(defaults);
      }
      executePrediction(false);
      showToast('Reset property sliders to benchmark baseline', 'success');
    });
  }

  // -------------------------------------------------------------
  // 7. Prediction Execution & UI Rendering
  // -------------------------------------------------------------
  async function executePrediction(isInitial = false) {
    const payload = collectFormValues();

    if (!isInitial) {
      predictSpinner.classList.remove('hidden');
    }

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Prediction calculation failed.');
      }

      renderPredictionResults(data);

      if (data.warnings && data.warnings.length > 0 && !isInitial) {
        data.warnings.forEach(w => showToast(w, 'warning'));
      }
    } catch (err) {
      console.error('Prediction error:', err);
      showToast(err.message, 'error');
    } finally {
      predictSpinner.classList.add('hidden');
    }
  }

  function renderPredictionResults(data) {
    const pred = data.prediction;

    // 1. Price formatting & Counter Animation
    const currentPriceText = priceDisplay.textContent.replace(/,/g, '');
    const currentPrice = parseFloat(currentPriceText) || 0;
    const targetPrice = pred.price_k * 1000;
    animateValue(priceDisplay, currentPrice, targetPrice, 350);

    adjustedPriceDisplay.textContent = `~${pred.adjusted_price_formatted}`;

    // 2. Tier Badge
    const cat = pred.category;
    tierBadge.textContent = cat.tier;
    tierBadge.className = `tier-badge ${cat.badge_class}`;

    // 3. Confidence Strip
    const conf = pred.confidence_interval;
    confidenceRangeDisplay.textContent = `${conf.lower_formatted} — ${conf.upper_formatted}`;

    // 4. Percentile Gauge
    const pct = Math.min(100, Math.max(0, pred.percentile));
    percentileBar.style.width = `${pct}%`;
    percentileMarker.style.left = `${pct}%`;
    
    let standingWord = 'Average';
    if (pct < 30) standingWord = 'Budget / High-Affordability';
    else if (pct < 60) standingWord = 'Balanced Mid-Market';
    else if (pct < 85) standingWord = 'Upper Tier';
    else standingWord = 'Top Tier Luxury';

    percentileRankText.textContent = `${standingWord} (${pct}% Percentile)`;

    // 5. Baseline Reference
    if (pred.baseline_comparison) {
      baselineRefText.textContent = `Baseline Avg: $${pred.baseline_comparison.baseline_k}k`;
    }

    // 6. Last Evaluated Tag
    const now = new Date();
    lastEvaluatedTag.textContent = `Live Synced: ${now.toLocaleTimeString()}`;

    // 7. Update Charts
    updateImpactChart(data.feature_impacts);
    updateRadarChart(data.normalized_profile);
  }

  function animateValue(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      const val = Math.floor(progress * (end - start) + start);
      element.textContent = val.toLocaleString();
      if (progress < 1) {
        window.requestAnimationFrame(step);
      } else {
        element.textContent = Math.round(end).toLocaleString();
      }
    };
    window.requestAnimationFrame(step);
  }

  // -------------------------------------------------------------
  // 8. Visualizations: Feature Impact Bar Chart & Radar Profile
  // -------------------------------------------------------------
  function updateImpactChart(impacts) {
    if (!impacts || !impacts.length) return;

    const ctx = document.getElementById('impactChart');
    if (!ctx) return;

    const themeColors = getThemeColors();
    // Pick top 7 impactful features for clean readability
    const topImpacts = impacts.slice(0, 7);

    const labels = topImpacts.map(i => `${i.feature} (${i.name.length > 20 ? i.name.slice(0, 18) + '...' : i.name})`);
    const values = topImpacts.map(i => i.impact_k);
    const colors = topImpacts.map(i => i.impact_k >= 0 ? '#10b981' : '#f43f5e');

    if (impactChart) {
      impactChart.data.labels = labels;
      impactChart.data.datasets[0].data = values;
      impactChart.data.datasets[0].backgroundColor = colors;
      impactChart.options.scales.x.grid.color = themeColors.gridColor;
      impactChart.options.scales.x.ticks.color = themeColors.textMuted;
      impactChart.options.scales.y.ticks.color = themeColors.textColor;
      impactChart.update();
      return;
    }

    impactChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Value Impact ($1,000s)',
          data: values,
          backgroundColor: colors,
          borderRadius: 6,
          borderSkipped: false
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => {
                const val = context.raw;
                const sign = val >= 0 ? '+' : '';
                return `Marginal Impact: ${sign}$${val}k (${sign}$${Math.round(val * 1000).toLocaleString()})`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: themeColors.gridColor },
            ticks: {
              color: themeColors.textMuted,
              callback: (val) => `${val >= 0 ? '+' : ''}$${val}k`
            }
          },
          y: {
            grid: { display: false },
            ticks: {
              color: themeColors.textColor,
              font: { size: 11, family: 'Inter' }
            }
          }
        }
      }
    });
  }

  function updateRadarChart(normalizedProfile) {
    if (!normalizedProfile) return;

    const ctx = document.getElementById('radarChart');
    if (!ctx) return;

    const themeColors = getThemeColors();
    const keys = Object.keys(normalizedProfile);
    const propertyValues = keys.map(k => normalizedProfile[k]);
    // Boston average normalized baseline (approx 50% relative)
    const baselineValues = keys.map(() => 50);

    if (radarChart) {
      radarChart.data.labels = keys;
      radarChart.data.datasets[0].data = propertyValues;
      radarChart.data.datasets[1].data = baselineValues;
      radarChart.options.scales.r.grid.color = themeColors.gridColor;
      radarChart.options.scales.r.pointLabels.color = themeColors.textColor;
      radarChart.update();
      return;
    }

    radarChart = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: keys,
        datasets: [
          {
            label: 'Selected Property',
            data: propertyValues,
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99, 102, 241, 0.25)',
            borderWidth: 2,
            pointBackgroundColor: '#6366f1'
          },
          {
            label: 'Boston Market Baseline',
            data: baselineValues,
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.08)',
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointBackgroundColor: '#06b6d4'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        elements: { line: { tension: 0.1 } },
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: themeColors.textColor, font: { size: 11 } }
          }
        },
        scales: {
          r: {
            angleLines: { color: themeColors.gridColor },
            grid: { color: themeColors.gridColor },
            suggestedMin: 0,
            suggestedMax: 100,
            ticks: { display: false, stepSize: 25 },
            pointLabels: {
              color: themeColors.textColor,
              font: { size: 10, weight: 600, family: 'JetBrains Mono' }
            }
          }
        }
      }
    });
  }

  // -------------------------------------------------------------
  // 9. Batch CSV Processing & Table Management
  // -------------------------------------------------------------
  function setupBatchUploader() {
    // Drop Zone Click
    batchDropZone.addEventListener('click', (e) => {
      if (e.target.closest('#removeFileBtn')) return;
      batchFileInput.click();
    });

    // File Input Change
    batchFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleBatchFileSelect(e.target.files[0]);
      }
    });

    // Drag and Drop
    ['dragenter', 'dragover'].forEach(eventName => {
      batchDropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        batchDropZone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      batchDropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        batchDropZone.classList.remove('dragover');
      });
    });

    batchDropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt.files && dt.files.length > 0) {
        handleBatchFileSelect(dt.files[0]);
      }
    });

    removeFileBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      resetBatchFile();
    });

    processBatchBtn.addEventListener('click', processBatchFile);
    batchTableSearch.addEventListener('input', () => {
      batchCurrentPage = 1;
      renderBatchTable();
    });

    prevPageBtn.addEventListener('click', () => {
      if (batchCurrentPage > 1) {
        batchCurrentPage--;
        renderBatchTable();
      }
    });

    nextPageBtn.addEventListener('click', () => {
      const filtered = getFilteredBatchResults();
      const totalPages = Math.ceil(filtered.length / batchPageSize);
      if (batchCurrentPage < totalPages) {
        batchCurrentPage++;
        renderBatchTable();
      }
    });

    exportCsvBtn.addEventListener('click', exportBatchResultsToCSV);
  }

  function handleBatchFileSelect(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      showToast('Please upload a valid .csv file.', 'error');
      return;
    }

    selectedBatchFile = file;
    chosenFileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    fileChosenBanner.classList.remove('hidden');
    batchDropZone.querySelector('.drop-zone-content').classList.add('hidden');
    processBatchBtn.disabled = false;
  }

  function resetBatchFile() {
    selectedBatchFile = null;
    batchFileInput.value = '';
    fileChosenBanner.classList.add('hidden');
    batchDropZone.querySelector('.drop-zone-content').classList.remove('hidden');
    processBatchBtn.disabled = true;
    batchResultsSection.classList.add('hidden');
  }

  async function processBatchFile() {
    if (!selectedBatchFile) return;

    processBatchBtn.disabled = true;
    batchSpinner.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', selectedBatchFile);

    try {
      const response = await fetch('/api/predict-batch', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Failed processing batch dataset.');
      }

      batchResultsData = data.results || [];
      renderBatchSummary(data.summary);
      batchCurrentPage = 1;
      renderBatchTable();
      batchResultsSection.classList.remove('hidden');
      showToast(`Scored ${data.summary.successful_predictions} properties successfully!`, 'success');
    } catch (err) {
      console.error('Batch error:', err);
      showToast(err.message, 'error');
    } finally {
      processBatchBtn.disabled = false;
      batchSpinner.classList.add('hidden');
    }
  }

  function renderBatchSummary(summary) {
    if (!summary) return;
    batchTotalCount.textContent = summary.successful_predictions;
    batchMeanVal.textContent = summary.mean_formatted;
    batchMinVal.textContent = `$${(summary.min_price_k * 1000).toLocaleString()}`;
    batchMaxVal.textContent = `$${(summary.max_price_k * 1000).toLocaleString()}`;
  }

  function getFilteredBatchResults() {
    const q = batchTableSearch.value.trim().toLowerCase();
    if (!q) return batchResultsData;

    return batchResultsData.filter(r => {
      if (r.status !== 'success') return false;
      const desc = (r.features && r.features.description) || '';
      const cat = r.category || '';
      const price = r.price_formatted || '';
      return desc.toLowerCase().includes(q) || cat.toLowerCase().includes(q) || price.includes(q);
    });
  }

  function renderBatchTable() {
    const filtered = getFilteredBatchResults();
    const totalRecords = filtered.length;
    const totalPages = Math.ceil(totalRecords / batchPageSize) || 1;

    batchTableBody.innerHTML = '';

    if (totalRecords === 0) {
      batchTableBody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding: 24px; color: var(--text-muted);">No matching property records found.</td></tr>`;
      pageInfo.textContent = 'No records';
      prevPageBtn.disabled = true;
      nextPageBtn.disabled = true;
      return;
    }

    const startIdx = (batchCurrentPage - 1) * batchPageSize;
    const endIdx = Math.min(startIdx + batchPageSize, totalRecords);
    const pageRecords = filtered.slice(startIdx, endIdx);

    pageRecords.forEach(r => {
      const tr = document.createElement('tr');
      const feat = r.features || {};
      const desc = feat.description || `Property #${r.row_index}`;

      tr.innerHTML = `
        <td><strong>#${r.row_index}</strong></td>
        <td><span style="font-weight: 600;">${desc}</span></td>
        <td><code>${feat.RM ? feat.RM.toFixed(1) : '-'}</code></td>
        <td><code>${feat.CRIM ? feat.CRIM.toFixed(2) : '-'}</code></td>
        <td><code>$${feat.TAX ? feat.TAX.toFixed(0) : '-'}</code></td>
        <td><strong class="text-gradient">${r.price_formatted}</strong></td>
        <td><span class="text-cyan">${r.adjusted_price_formatted}</span></td>
        <td><span class="tier-badge ${getTierBadgeClass(r.category)}">${r.category}</span></td>
        <td>${r.percentile}% Pct</td>
      `;
      batchTableBody.appendChild(tr);
    });

    pageInfo.textContent = `Showing ${startIdx + 1} to ${endIdx} of ${totalRecords} records`;
    prevPageBtn.disabled = batchCurrentPage <= 1;
    nextPageBtn.disabled = batchCurrentPage >= totalPages;
  }

  function getTierBadgeClass(tier) {
    if (!tier) return 'badge-moderate';
    if (tier.includes('Affordable')) return 'badge-affordable';
    if (tier.includes('Mid-Market')) return 'badge-moderate';
    if (tier.includes('Upper')) return 'badge-upper';
    if (tier.includes('Premium')) return 'badge-premium';
    return 'badge-luxury';
  }

  async function exportBatchResultsToCSV() {
    if (!batchResultsData || batchResultsData.length === 0) {
      showToast('No batch predictions available to export.', 'warning');
      return;
    }

    try {
      const response = await fetch('/api/export-results', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ results: batchResultsData })
      });

      if (!response.ok) throw new Error('Failed exporting batch predictions.');

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `boston_housing_predictions_${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      showToast('Exported predictions CSV successfully!', 'success');
    } catch (err) {
      console.error('Export error:', err);
      showToast(err.message, 'error');
    }
  }

  // -------------------------------------------------------------
  // 10. Code Snippets & Copy Helper
  // -------------------------------------------------------------
  function setupCodeSnippets() {
    const snippetContainers = document.querySelectorAll('.code-snippets-container');

    snippetContainers.forEach(container => {
      const tabs = container.querySelectorAll('.code-tab');
      const blocks = container.querySelectorAll('.code-block');
      const copyBtn = container.querySelector('.btn-copy-code');

      tabs.forEach(tab => {
        tab.addEventListener('click', () => {
          tabs.forEach(t => t.classList.remove('active'));
          tab.classList.add('active');

          const lang = tab.getAttribute('data-lang');
          blocks.forEach(block => {
            if (block.getAttribute('data-lang-block') === lang) {
              block.classList.remove('hidden');
            } else {
              block.classList.add('hidden');
            }
          });
        });
      });

      if (copyBtn) {
        copyBtn.addEventListener('click', () => {
          const visibleBlock = container.querySelector('.code-block:not(.hidden)');
          if (visibleBlock) {
            navigator.clipboard.writeText(visibleBlock.textContent.trim()).then(() => {
              showToast('Code copied to clipboard!', 'success');
            }).catch(() => {
              showToast('Failed to copy code.', 'error');
            });
          }
        });
      }
    });
  }

  // -------------------------------------------------------------
  // 11. Toast Notifications
  // -------------------------------------------------------------
  function showToast(message, type = 'info') {
    if (!toastContainer) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    let iconClass = 'fa-solid fa-circle-info';
    if (type === 'success') iconClass = 'fa-solid fa-circle-check';
    else if (type === 'error') iconClass = 'fa-solid fa-triangle-exclamation';
    else if (type === 'warning') iconClass = 'fa-solid fa-circle-exclamation';

    toast.innerHTML = `<i class="${iconClass}"></i><span>${message}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(100%)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

})();
