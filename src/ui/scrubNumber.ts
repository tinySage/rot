import { paramStore, F_RANGE, K_RANGE } from '../state';

type Param = 'F' | 'k';

const RANGES: Record<Param, { min: number; max: number }> = { F: F_RANGE, k: K_RANGE };
const DECIMALS: Record<Param, number> = { F: 3, k: 4 };
const STEP: Record<Param, number> = { F: 0.001, k: 0.0005 };

// Full horizontal drag, in pixels, mapped to the whole parameter range.
const DRAG_PX_FOR_FULL_RANGE = 480;

function formatValue(param: Param, value: number): string {
  return value.toFixed(DECIMALS[param]);
}

/** Enhances every `[data-scrub]` element into a Bret-Victor-style draggable number. */
export function enhanceScrubNumbers(root: ParentNode): void {
  root.querySelectorAll<HTMLElement>('[data-scrub]').forEach((el) => {
    const param = el.dataset.scrub as Param;
    if (param !== 'F' && param !== 'k') return;
    const range = RANGES[param];

    el.classList.add('scrub');
    el.tabIndex = 0;
    el.setAttribute('role', 'slider');
    el.setAttribute('aria-valuemin', String(range.min));
    el.setAttribute('aria-valuemax', String(range.max));

    paramStore.subscribe((p) => {
      const value = param === 'F' ? p.F : p.k;
      el.textContent = formatValue(param, value);
      el.setAttribute('aria-valuenow', value.toFixed(DECIMALS[param]));
    });

    let dragging = false;
    let startX = 0;
    let startValue = 0;

    const setParam = (value: number) => {
      if (param === 'F') paramStore.setF(value);
      else paramStore.setK(value);
    };

    el.addEventListener('pointerdown', (event) => {
      dragging = true;
      startX = event.clientX;
      startValue = param === 'F' ? paramStore.F : paramStore.k;
      el.setPointerCapture(event.pointerId);
      el.classList.add('scrubbing');
    });

    el.addEventListener('pointermove', (event) => {
      if (!dragging) return;
      const deltaPx = event.clientX - startX;
      const span = range.max - range.min;
      setParam(startValue + (deltaPx / DRAG_PX_FOR_FULL_RANGE) * span);
    });

    const endDrag = (event: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      el.classList.remove('scrubbing');
      try {
        el.releasePointerCapture(event.pointerId);
      } catch {
        // pointer capture may already be released
      }
    };
    el.addEventListener('pointerup', endDrag);
    el.addEventListener('pointercancel', endDrag);

    el.addEventListener('keydown', (event) => {
      const current = param === 'F' ? paramStore.F : paramStore.k;
      if (event.key === 'ArrowRight' || event.key === 'ArrowUp') {
        event.preventDefault();
        setParam(current + STEP[param]);
      } else if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') {
        event.preventDefault();
        setParam(current - STEP[param]);
      }
    });
  });
}
