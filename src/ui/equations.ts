import katex from 'katex';

export function renderEquation(elementId: string, tex: string, displayMode = true): void {
  const el = document.getElementById(elementId);
  if (!el) return;
  katex.render(tex, el, {
    throwOnError: false,
    strict: false,
    trust: true,
    displayMode,
  });
}

type Param = 'F' | 'k';

const SYMBOL_CLASS: Record<Param, string> = { F: 'eqsym-F', k: 'eqsym-k' };

function setHighlight(param: Param, active: boolean): void {
  document.body.classList.toggle(`highlight-${param}`, active);
  document.querySelectorAll(`[data-param="${param}"]`).forEach((el) => {
    el.classList.toggle('highlight', active);
  });
}

/** Wires hover on `.eqsym-F` / `.eqsym-k` spans (set via KaTeX's \class{}) to highlight
 *  the matching slider and phase-map guide line. */
export function wireEquationHover(root: ParentNode = document): void {
  (Object.keys(SYMBOL_CLASS) as Param[]).forEach((param) => {
    root.querySelectorAll<HTMLElement>(`.${SYMBOL_CLASS[param]}`).forEach((el) => {
      el.addEventListener('mouseenter', () => setHighlight(param, true));
      el.addEventListener('mouseleave', () => setHighlight(param, false));
      el.addEventListener('focus', () => setHighlight(param, true));
      el.addEventListener('blur', () => setHighlight(param, false));
    });
  });
}
