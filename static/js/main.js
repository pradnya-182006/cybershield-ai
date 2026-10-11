const root = document.documentElement;
const btn = document.getElementById('themeBtn');
const saved = localStorage.getItem('theme') || 'dark';
root.setAttribute('data-theme', saved);
btn.textContent = saved === 'dark' ? '🌙' : '☀️';
btn.addEventListener('click', () => {
  const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  root.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  btn.textContent = next === 'dark' ? '🌙' : '☀️';
});