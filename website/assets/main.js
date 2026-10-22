const navigation = document.querySelector('.section-navigation');
const wideScreen = matchMedia('(min-width: 1680px)');
const syncNavigation = () => { navigation.open = wideScreen.matches; };
navigation.dataset.enhanced = '';
syncNavigation();
wideScreen.addEventListener('change', syncNavigation);
navigation.querySelectorAll('a').forEach(link => {
  link.addEventListener('click', () => { if (!wideScreen.matches) navigation.open = false; });
});
document.addEventListener('click', event => {
  if (!wideScreen.matches && !navigation.contains(event.target)) navigation.open = false;
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && navigation.open && !wideScreen.matches) {
    const focusedInside = navigation.contains(document.activeElement);
    navigation.open = false;
    if (focusedInside) navigation.querySelector('summary').focus();
  }
});
const sections = [...document.querySelectorAll('main section[id]')];
const links = [...navigation.querySelectorAll('a')];
let scheduled = false;
const updateActiveSection = () => {
  let active = sections[0];
  for (const section of sections) {
    if (section.getBoundingClientRect().top <= innerHeight * 0.3) active = section;
  }
  for (const link of links) {
    if (link.hash === '#' + active.id) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  }
  scheduled = false;
};
addEventListener('scroll', () => {
  if (!scheduled) { scheduled = true; requestAnimationFrame(updateActiveSection); }
}, { passive: true });
updateActiveSection();
const copyButton = document.querySelector('#copy-citation');
copyButton.hidden = false;
copyButton.addEventListener('click', async () => {
  const citation = document.querySelector('#bibtex');
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(citation.textContent.trim());
    status.textContent = 'Citation copied.';
    copyButton.textContent = 'Copied';
    setTimeout(() => { copyButton.textContent = 'Copy BibTeX'; status.textContent = ''; }, 2500);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = getSelection();
    selection.removeAllRanges(); selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
