const routes = {
  direct: { title: 'A shorter route for a direct question.', description: 'The controller could skip work that contributes little to a straightforward answer.', states: ['on','on','skip','on','skip','on','skip','on'] },
  multistep: { title: 'More steps when the question needs them.', description: 'For harder reasoning, the controller could execute a deeper sequence instead of cutting layers.', states: ['on','on','on','on','on','on','on','on'] },
  revisit: { title: 'Revisit a useful transformation.', description: 'The controller could repeat a layer when another pass helps refine the intermediate state.', states: ['on','on','on','loop','on','skip','on','on'] }
};

const stage = document.querySelector('.depth-stage');
for (let i = 0; i < 8; i += 1) {
  const layer = document.createElement('div');
  layer.className = 'model-layer';
  layer.style.setProperty('--i', i);
  layer.innerHTML = `<span class="layer-label">L${String(i + 1).padStart(2, '0')}</span><span class="layer-lines"><i></i><i></i><i></i></span><span class="status-dot"></span>`;
  stage.appendChild(layer);
}

function selectRoute(name) {
  const route = routes[name];
  document.querySelector('#route-title').textContent = route.title;
  document.querySelector('#route-description').textContent = route.description;
  document.querySelectorAll('.model-layer').forEach((layer, i) => {
    layer.classList.remove('on', 'skip', 'loop');
    layer.classList.add(route.states[i]);
  });
  document.querySelectorAll('.route-button').forEach((button) => {
    const selected = button.dataset.route === name;
    button.classList.toggle('active', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
}
document.querySelectorAll('.route-button').forEach((button) => button.addEventListener('click', () => selectRoute(button.dataset.route)));
selectRoute('direct');

const menuButton = document.querySelector('.menu-toggle');
const menu = document.querySelector('.mobile-menu');
function closeMenu() { menuButton.setAttribute('aria-expanded', 'false'); document.body.classList.remove('menu-open'); }
menuButton.addEventListener('click', () => { const open = menuButton.getAttribute('aria-expanded') !== 'true'; menuButton.setAttribute('aria-expanded', String(open)); document.body.classList.toggle('menu-open', open); });
menu.querySelectorAll('a').forEach((link) => link.addEventListener('click', closeMenu));
document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeMenu(); });
window.addEventListener('resize', () => { if (window.innerWidth > 760) closeMenu(); });

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
if (!reducedMotion.matches) {
  const observer = new IntersectionObserver((entries) => entries.forEach((entry) => { if (entry.isIntersecting) { entry.target.classList.add('visible'); observer.unobserve(entry.target); } }), { threshold: 0.14 });
  document.querySelectorAll('.reveal').forEach((element) => observer.observe(element));
  let ticking = false;
  function updateTilt() {
    const showcase = document.querySelector('.route-showcase');
    const rect = showcase.getBoundingClientRect();
    const progress = Math.max(0, Math.min(1, (window.innerHeight - rect.top) / (window.innerHeight + rect.height)));
    stage.style.setProperty('--scroll-tilt', `${-6 + progress * 12}deg`);
    ticking = false;
  }
  window.addEventListener('scroll', () => { if (!ticking) { requestAnimationFrame(updateTilt); ticking = true; } }, { passive: true });
  updateTilt();
} else {
  document.querySelectorAll('.reveal').forEach((element) => element.classList.add('visible'));
}
