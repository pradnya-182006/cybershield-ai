let charts = [];

function chart(id, type, data, opts = {}) {
  const c = new Chart(document.getElementById(id), {
    type, data, options: {plugins: {legend: {labels: {color: '#93A0C2'}}}, ...opts}
  });
  charts.push(c);
}

async function load() {
  charts.forEach(c => c.destroy()); charts = [];
  const s = await (await fetch('/api/stats')).json();
  document.getElementById('kTotal').textContent = s.total;
  document.getElementById('kToxic').textContent = s.toxic_pct + '%';
  document.getElementById('kHigh').textContent = s.high_count;
  document.getElementById('kSafe').textContent = s.safe_pct + '%';

  const axis = {scales: {x: {ticks: {color: '#93A0C2'}}, y: {ticks: {color: '#93A0C2', precision: 0}, beginAtZero: true}}};
  chart('trendChart', 'line', {
    labels: s.trend.labels,
    datasets: [
      {label: 'Total', data: s.trend.total, borderColor: '#6366F1', tension: .4},
      {label: 'Harmful', data: s.trend.harmful, borderColor: '#F43F5E', tension: .4}
    ]}, axis);
  const cats = Object.keys(s.categories);
  chart('catChart', 'doughnut', {
    labels: cats.length ? cats : ['No data'],
    datasets: [{data: cats.length ? Object.values(s.categories) : [1],
      backgroundColor: ['#6366F1', '#22D3EE', '#F43F5E', '#F59E0B', '#10B981', '#A855F7']}]});
  chart('sevChart', 'bar', {
    labels: Object.keys(s.severity),
    datasets: [{label: 'Comments', data: Object.values(s.severity),
      backgroundColor: ['#10B981', '#22D3EE', '#F59E0B', '#F43F5E']}]}, axis);

  const h = await (await fetch('/api/history')).json();
  document.querySelector('#table tbody').innerHTML = h.map(r =>
    `<tr><td>${r.time}</td><td>${r.text.replace(/</g, '&lt;')}</td><td>${r.label.replace('_', ' ')}</td>
     <td>${r.severity}</td><td>${Math.round(r.score * 100)}%</td></tr>`).join('') ||
    '<tr><td colspan="5">No analyses yet. Try the Analyzer!</td></tr>';
}

document.getElementById('clearHist').addEventListener('click', async () => {
  if (confirm('Clear all history?')) { await fetch('/api/history', {method: 'DELETE'}); load(); }
});
load();