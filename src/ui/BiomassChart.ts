export interface BiomassSample {
  t: number;
  v: number;
}

/**
 * Hand-drawn canvas line chart, no charting library. Axis labels are set in
 * IBM Plex Mono per CLAUDE.md; colours are read live from the page's CSS
 * custom properties so the chart follows light/dark mode.
 */
export class BiomassChart {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private dpr = 1;
  private cssWidth = 0;
  private cssHeight = 0;
  private samples: BiomassSample[] = [];

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  setSamples(samples: BiomassSample[]): void {
    this.samples = samples;
    this.draw();
  }

  private resize(): void {
    const rect = this.canvas.getBoundingClientRect();
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.cssWidth = rect.width;
    this.cssHeight = rect.height;
    const w = Math.round(this.cssWidth * this.dpr);
    const h = Math.round(this.cssHeight * this.dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.draw();
  }

  private draw(): void {
    const { ctx } = this;
    const w = this.cssWidth;
    const h = this.cssHeight;
    if (w === 0 || h === 0) return;

    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const style = getComputedStyle(document.documentElement);
    const ruleColor = style.getPropertyValue('--rule').trim() || '#d9d4c8';
    const inkColor = style.getPropertyValue('--ink').trim() || '#1c1b19';
    const mutedColor = style.getPropertyValue('--muted').trim() || '#6b6760';
    const accentColor = style.getPropertyValue('--accent').trim() || '#9a3412';

    const padLeft = 50;
    const padRight = 12;
    const padTop = 12;
    const padBottom = 26;
    const plotW = Math.max(1, w - padLeft - padRight);
    const plotH = Math.max(1, h - padTop - padBottom);

    ctx.font = '11px "IBM Plex Mono", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';

    if (this.samples.length < 2) {
      ctx.fillStyle = mutedColor;
      ctx.fillText('collecting samples…', padLeft, padTop + plotH / 2);
      return;
    }

    const minT = this.samples[0].t;
    const maxT = this.samples[this.samples.length - 1].t;
    const maxV = Math.max(...this.samples.map((s) => s.v), 1);

    const xScale = (t: number) => padLeft + ((t - minT) / Math.max(maxT - minT, 1e-6)) * plotW;
    const yScale = (v: number) => padTop + plotH - (v / maxV) * plotH;

    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const v = (maxV / yTicks) * i;
      const y = yScale(v);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.strokeStyle = ruleColor;
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = mutedColor;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(formatBiomass(v), padLeft - 8, y);
    }

    const xTicks = Math.min(5, this.samples.length - 1);
    for (let i = 0; i <= xTicks; i++) {
      const t = minT + ((maxT - minT) / xTicks) * i;
      const x = xScale(t);
      ctx.fillStyle = mutedColor;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(`${t.toFixed(0)}s`, x, padTop + plotH + 8);
    }

    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.strokeStyle = inkColor;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.beginPath();
    this.samples.forEach((s, i) => {
      const x = xScale(s.t);
      const y = yScale(s.v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = accentColor;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

function formatBiomass(v: number): string {
  if (v >= 1000) return `${(v / 1000).toFixed(1)}k`;
  return v.toFixed(0);
}
