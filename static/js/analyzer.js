const input = document.getElementById('text');
const result = document.getElementById('result');
const analyzeBtn = document.getElementById('analyzeBtn');
let lastRewrite = '';

const esc = s => s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

input.addEventListener('input', () => {
  document.getElementById('counter').textContent = `${input.value.length} / 1000`;
});
document.querySelectorAll('.chip[data-t]').forEach(c =>
  c.addEventListener('click', () => { input.value = c.dataset.t; input.dispatchEvent(new Event('input')); }));
document.getElementById('clearBtn').addEventListener('click', () => {
  input.value = ''; result.classList.add('hidden'); input.dispatchEvent(new Event('input'));
});
analyzeBtn.addEventListener('click', analyze);
document.getElementById('copyBtn').addEventListener('click', () => {
  navigator.clipboard.writeText(lastRewrite);
  document.getElementById('copyBtn').textContent = 'Copied!';
  setTimeout(() => document.getElementById('copyBtn').textContent = 'Copy', 1500);
});
document.getElementById('useBtn').addEventListener('click', () => { input.value = lastRewrite; input.dispatchEvent(new Event('input')); analyze(); });

function highlight(text, flagged) {
  const map = {};
  flagged.forEach(f => map[f.word.toLowerCase()] = f);
  return text.split(/(\b\w+\b)/).map(p => {
    const f = map[p.toLowerCase()];
    return f ? `<mark title="${esc(f.reason)}">${esc(p)}</mark>` : esc(p);
  }).join('');
}

async function analyze() {
  const text = input.value.trim();
  if (!text) return;
  analyzeBtn.disabled = true; analyzeBtn.textContent = 'Scanning...';
  try {
    const res = await fetch('/api/analyze', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})
    });
    const d = await res.json();
    if (!res.ok) { alert(d.error); } else { render(d, text); }
  } catch (e) { alert('Server error. Is Flask running?'); }
  analyzeBtn.disabled = false; analyzeBtn.textContent = 'Analyze';
}

function render(d, text) {
  result.classList.remove('hidden');
  const pct = Math.round(d.overall_score * 100);
  const color = d.severity === 'high' ? 'var(--bad)' : d.severity === 'medium' ? 'var(--warn)' : 'var(--ok)';
  const g = document.getElementById('gauge');
  g.style.setProperty('--c', color); g.style.setProperty('--p', pct);
  document.getElementById('gaugeVal').textContent = pct + '%';

  document.getElementById('verdict').textContent =
    d.label === 'safe' ? '✅ Safe' : (d.severity === 'high' || d.label === 'threat') ? '🚨 Cyberbullying Detected' : '⚠️ Potentially Harmful';
  const sev = document.getElementById('severity');
  sev.textContent = d.severity === 'none' ? 'No severity' : d.severity.toUpperCase() + ' severity';
  sev.className = 'sev ' + d.severity;
  document.getElementById('tags').innerHTML = d.tone_tags.map(t => `<span class="chip tag">${t}</span>`).join('');

  document.getElementById('bars').innerHTML = Object.entries(d.scores).map(([k, v]) =>
    `<div class="bar"><div class="bar-top"><span>${k.replace('_',' ')}</span><span>${Math.round(v*100)}%</span></div>
     <div class="track"><div class="fill" style="width:${v*100}%"></div></div></div>`).join('');

  document.getElementById('highlighted').innerHTML = highlight(text, d.flagged_words);
  document.getElementById('explanation').textContent = d.explanation;
  lastRewrite = d.safer_rewrite;
  document.getElementById('rewrite').textContent = d.safer_rewrite;
  document.getElementById('disclaimer').textContent = d.disclaimer;
  result.scrollIntoView({behavior: 'smooth'});
}