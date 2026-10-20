/* CyberShield AI - Analyzer + Agentic Pipeline JS */
'use strict';

const input      = document.getElementById('text');
const resultDiv  = document.getElementById('result');
const analyzeBtn = document.getElementById('analyzeBtn');
const agentBtn   = document.getElementById('agentBtn');
const agentPanel = document.getElementById('agentPanel');
let lastRewrite  = '';

/* ── Helpers ─────────────────────────────────────────────────────────── */
const esc = s => s.replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function scoreColor(label, score) {
  if (label === 'safe') return 'ok';
  if (label === 'threat') return score > 0.35 ? 'crit' : 'muted';
  if (label === 'obscene') return score > 0.40 ? 'warn' : 'muted';
  if (label === 'sarcasm') return 'muted';
  return score > 0.40 ? 'bad' : 'muted';
}

/* ── Counter & chips ─────────────────────────────────────────────────── */
input.addEventListener('input', () => {
  document.getElementById('counter').textContent = `${input.value.length} / 1000`;
});
document.querySelectorAll('.chip[data-t]').forEach(c =>
  c.addEventListener('click', () => {
    input.value = c.dataset.t;
    input.dispatchEvent(new Event('input'));
  }));
document.getElementById('clearBtn').addEventListener('click', () => {
  input.value = '';
  resultDiv.classList.add('hidden');
  agentPanel.classList.add('hidden');
  input.dispatchEvent(new Event('input'));
});

/* ── Copy / Use ──────────────────────────────────────────────────────── */
analyzeBtn.addEventListener('click', runAnalyze);
agentBtn.addEventListener('click', runAgent);

document.getElementById('copyBtn').addEventListener('click', () => {
  navigator.clipboard.writeText(lastRewrite);
  const btn = document.getElementById('copyBtn');
  btn.textContent = 'Copied!';
  setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
});
document.getElementById('useBtn').addEventListener('click', () => {
  input.value = lastRewrite;
  input.dispatchEvent(new Event('input'));
  runAgent();
});

/* ── Highlight flagged words ─────────────────────────────────────────── */
function highlight(text, flagged) {
  const map = {};
  flagged.forEach(f => map[f.word.toLowerCase()] = f);
  return text.split(/(\b\w+\b)/).map(p => {
    const f = map[p.toLowerCase()];
    return f ? `<mark title="${esc(f.reason)}">${esc(p)}</mark>` : esc(p);
  }).join('');
}

/* ── Standard Analyze ────────────────────────────────────────────────── */
async function runAnalyze() {
  const text = input.value.trim();
  if (!text) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Scanning...';
  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const d = await res.json();
    if (!res.ok) { alert(d.error); }
    else { renderResult(d, text); }
  } catch (e) { alert('Server error. Is Flask running?'); }
  analyzeBtn.disabled = false;
  analyzeBtn.textContent = '\uD83D\uDD0D Analyze';
}

/* ── Agent Pipeline ──────────────────────────────────────────────────── */
async function runAgent() {
  const text = input.value.trim();
  if (!text) return;

  agentBtn.disabled = true;
  agentBtn.textContent = 'Running Pipeline...';
  analyzeBtn.disabled = true;

  // Show panel immediately with pending stages
  agentPanel.classList.remove('hidden');
  document.getElementById('agentSpinner').classList.add('active');
  document.getElementById('agentBar').classList.add('active');
  document.getElementById('agentSummaryBar').classList.add('hidden');
  renderPendingStages();

  try {
    const res = await fetch('/api/agent', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const d = await res.json();
    if (!res.ok) {
      alert(d.error || 'Agent pipeline error.');
    } else {
      renderResult(d.analysis, text);
      renderAgentStages(d.agent);
    }
  } catch (e) {
    alert('Server error. Is Flask running?');
  }

  document.getElementById('agentSpinner').classList.remove('active');
  document.getElementById('agentBar').classList.remove('active');
  agentBtn.disabled = false;
  agentBtn.textContent = '\uD83E\uDDE0 Run Agent Pipeline';
  analyzeBtn.disabled = false;
}

/* ── Render standard result ──────────────────────────────────────────── */
function renderResult(d, text) {
  resultDiv.classList.remove('hidden');

  // 1. Calculate Harmfulness Score Percentage
  const pct = Math.round(d.harmfulness_score * 100);

  // 2. Calculate Label Confidence Percentage
  let labelProb = 0;
  if (typeof d.label_confidence === 'number') {
    labelProb = d.label_confidence;
  } else if (d.scores && d.label && d.label in d.scores) {
    labelProb = d.scores[d.label];
  } else if (d.scores && d.scores.safe !== undefined && (d.label === 'safe' || d.label === 'uncertain')) {
    labelProb = d.scores.safe;
  } else if (d.scores) {
    // Fallback for legacy models where 'safe' is not a separate classifier weight
    const harmfulVals = Object.keys(d.scores).filter(k => k !== 'safe').map(k => d.scores[k]);
    const maxHarmful = harmfulVals.length ? Math.max(...harmfulVals) : 0;
    labelProb = Math.max(0, 1 - maxHarmful);
  }
  const labelConfPct = Math.round(labelProb * 100);

  // 3. Determine Decision State: Safe, Inconclusive/Uncertain, or Harmful
  const isUncertain = Boolean(
    d.uncertain ||
    d.label === 'uncertain' ||
    (d.label === 'safe' && (labelProb < 0.35 || (d.scores && d.scores.safe !== undefined && d.scores.safe < 0.35)))
  );

  let verdictText = '';
  let color = 'var(--ok)';
  let sevText = '';
  let sevClass = '';

  if (isUncertain) {
    // State: Inconclusive / Low Confidence
    verdictText = 'ℹ️ Inconclusive / Low Confidence';
    color = '#93A0C2'; // Neutral informational color
    sevText = 'Low confidence (No action taken)';
    sevClass = 'sev uncertain';
  } else if (d.label === 'safe') {
    // State: Confident Safe
    verdictText = '✅ Safe';
    color = 'var(--ok)';
    sevText = 'No severity';
    sevClass = 'sev none';
  } else {
    // State: Harmful Classification
    color = d.severity === 'high' ? 'var(--bad)' :
            d.severity === 'medium' ? 'var(--warn)' : 'var(--warn)';

    if (d.label === 'threat' || d.severity === 'high') {
      verdictText = d.label === 'threat' ? '🚨 Threat Detected' : '🚨 Cyberbullying Detected';
    } else if (d.label === 'sarcasm') {
      verdictText = '⚠️ Sarcasm Detected';
    } else if (d.label === 'insult') {
      verdictText = '⚠️ Insult Detected';
    } else if (d.label === 'obscene') {
      verdictText = '⚠️ Obscene Language Detected';
    } else if (d.label === 'toxic') {
      verdictText = '⚠️ Toxic Language Detected';
    } else {
      verdictText = '⚠️ Potentially Harmful';
    }

    sevText = d.severity.toUpperCase() + ' severity';
    sevClass = 'sev ' + d.severity;
  }

  const g = document.getElementById('gauge');
  g.style.setProperty('--c', color);
  g.style.setProperty('--p', pct);
  document.getElementById('gaugeVal').textContent = pct + '%';

  const labelConfEl = document.getElementById('labelConfVal');
  if (labelConfEl) {
    labelConfEl.textContent = labelConfPct + '%';
  }

  const harmScoreEl = document.getElementById('harmScoreVal');
  if (harmScoreEl) {
    harmScoreEl.textContent = pct + '%';
    harmScoreEl.style.color = color;
  }

  document.getElementById('verdict').textContent = verdictText;

  const sev = document.getElementById('severity');
  sev.textContent = sevText;
  sev.className = sevClass;

  document.getElementById('tags').innerHTML = (d.tone_tags || [])
    .map(t => `<span class="tone-tag">${esc(t)}</span>`).join('');

  // Score bars
  const labelOrder = ['safe','toxic','insult','threat','obscene','sarcasm'];
  const scoreKeys = labelOrder.filter(k => k in d.scores);
  const otherKeys = Object.keys(d.scores).filter(k => !labelOrder.includes(k));
  const allKeys = [...scoreKeys, ...otherKeys];

  document.getElementById('bars').innerHTML = allKeys.map(k => {
    const raw = d.scores[k] || 0;
    const v   = (d.label === 'safe' && k !== 'safe') ? 0 : raw;
    const cls = scoreColor(k, raw);
    return `<div class="bar">
      <div class="bar-top">
        <span class="bar-label">${esc(k.replace('_',' '))}</span>
        <span class="bar-val">${Math.round(v*100)}%</span>
      </div>
      <div class="track"><div class="fill ${cls}" style="width:${v*100}%"></div></div>
    </div>`;
  }).join('');

  // Predicted label tags
  const pl = document.getElementById('predicted-labels');
  if (d.predicted_labels && d.predicted_labels.length) {
    pl.innerHTML = d.predicted_labels.map(l =>
      `<span class="label-tag ${esc(l)}">${esc(l)}</span>`
    ).join('');
  } else {
    pl.innerHTML = '';
  }

  document.getElementById('highlighted').innerHTML =
    highlight(text, d.flagged_words || []);
  document.getElementById('explanation').textContent = d.explanation || '';
  lastRewrite = d.safer_rewrite || text;
  document.getElementById('rewrite').textContent = lastRewrite;
  document.getElementById('disclaimer').textContent = d.disclaimer || '';

  resultDiv.scrollIntoView({behavior:'smooth'});
}

/* ── Agent Stage Rendering ───────────────────────────────────────────── */
const STAGE_EMOJIS = {
  'DETECT':  '\uD83D\uDD0E',
  'ANALYZE': '\uD83E\uDDE0',
  'DECIDE':  '\u2696\uFE0F',
  'ACT':     '\u26A1',
};

function renderPendingStages() {
  const stages = [
    {stage:'DETECT',  stage_index:1, title:'Signal Detection',  summary:'Scanning for harmful patterns...'},
    {stage:'ANALYZE', stage_index:2, title:'Deep Analysis',      summary:'Reasoning over evidence...'},
    {stage:'DECIDE',  stage_index:3, title:'Policy Decision',    summary:'Selecting moderation action...'},
    {stage:'ACT',     stage_index:4, title:'Moderation Actions', summary:'Generating action steps...'},
  ];
  document.getElementById('stageList').innerHTML =
    stages.map(s => stageHtml(s, 'pending')).join('');
}

function renderAgentStages(agent) {
  const list = document.getElementById('stageList');
  list.innerHTML = (agent.stages || []).map(s => stageHtml(s, 'done', true)).join('');

  // Attach toggle listeners
  list.querySelectorAll('.stage-header').forEach(btn => {
    btn.addEventListener('click', () => {
      const expanded = btn.getAttribute('aria-expanded') === 'true';
      btn.setAttribute('aria-expanded', String(!expanded));
      const body = btn.nextElementSibling;
      if (body) body.classList.toggle('open', !expanded);
    });
  });

  // Summary bar
  const bar = document.getElementById('agentSummaryBar');
  bar.classList.remove('hidden');
  const badge = document.getElementById('agentFinalBadge');
  badge.innerHTML = `<span class="action-badge ${agent.final_badge_color || 'muted'}">${esc(agent.final_badge || 'UNKNOWN')}</span>`;
  document.getElementById('agentFinalDesc').textContent = agent.moderation_policy?.reason || '';
  document.getElementById('agentTiming').textContent = `Pipeline completed in ${agent.total_pipeline_ms}ms`;
  document.getElementById('agentDisclaimer').textContent = agent.disclaimer || '';
}

function stageHtml(stage, state, withBody) {
  const emoji = STAGE_EMOJIS[stage.stage] || '\u25B6';
  const timing = stage.duration_ms != null ? `${stage.duration_ms}ms` : '';
  const bodyHtml = withBody ? buildStageBody(stage) : '';

  return `<div class="stage-step ${state}" id="stage-${stage.stage_index}">
    <button class="stage-header" aria-expanded="false">
      <span class="stage-num">${stage.stage_index}</span>
      <span class="stage-emoji">${emoji}</span>
      <span class="stage-info">
        <div class="stage-title">${esc(stage.title)}</div>
        <div class="stage-summary">${esc(stage.summary)}</div>
      </span>
      ${timing ? `<span class="stage-timing">${timing}</span>` : ''}
      <span class="stage-chevron">&#9658;</span>
    </button>
    ${bodyHtml ? `<div class="stage-body">${bodyHtml}</div>` : ''}
  </div>`;
}

function buildStageBody(stage) {
  let html = '';

  // Reasoning list
  if (stage.reasoning && stage.reasoning.length) {
    html += `<ul class="reasoning-list" style="margin-bottom:10px">
      ${stage.reasoning.map(r => `<li>${esc(r)}</li>`).join('')}
    </ul>`;
  }

  // Evidence (DETECT)
  if (stage.evidence && stage.evidence.length) {
    html += `<div style="margin-bottom:8px">
      ${stage.evidence.slice(0,6).map(e => `<div class="concern-item">${esc(e)}</div>`).join('')}
    </div>`;
  }

  // Concerns (ANALYZE)
  if (stage.concerns && stage.concerns.length) {
    html += stage.concerns.map(c => `<div class="concern-item">${esc(c)}</div>`).join('');
  }
  if (stage.observations && stage.observations.length) {
    html += stage.observations.map(o => `<div class="obs-item">${esc(o)}</div>`).join('');
  }

  // Action badge (DECIDE)
  if (stage.action_badge) {
    html += `<span class="action-badge ${stage.action_badge_color || 'muted'}">${esc(stage.action_badge)}</span>`;
    if (stage.action_description) {
      html += `<p style="font-size:.82rem;color:var(--tx2);margin:4px 0 8px">${esc(stage.action_description)}</p>`;
    }
    if (stage.routing_factors && stage.routing_factors.length) {
      html += `<ul class="reasoning-list">
        ${stage.routing_factors.map(f => `<li>${esc(f)}</li>`).join('')}
      </ul>`;
    }
  }

  // Human review banner (DECIDE / ACT)
  if (stage.human_review_required) {
    html += `<div class="human-review-banner">
      &#128680; Human Moderator Review Required &mdash; Do Not Auto-Ban
    </div>`;
  }

  // Actions taken (ACT)
  if (stage.actions_taken && stage.actions_taken.length) {
    html += `<div style="margin-bottom:8px">
      ${stage.actions_taken.map(a =>
        `<div class="action-item">
          <span class="action-check">&#10003;</span>
          <span>${esc(a.action)}</span>
        </div>`
      ).join('')}
    </div>`;
  }

  // Recommendations (ACT)
  if (stage.recommendations && stage.recommendations.length) {
    html += `<div>
      <div style="font-size:.78rem;color:var(--mut);margin-bottom:5px;font-weight:600">Recommendations</div>
      ${stage.recommendations.map(r => {
        const isCrit = r.toUpperCase().startsWith('CRITICAL');
        return `<div class="rec-item ${isCrit ? 'critical' : ''}">${esc(r)}</div>`;
      }).join('')}
    </div>`;
  }

  return html;
}