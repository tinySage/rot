export interface Range {
  min: number;
  max: number;
}

export const F_RANGE: Range = { min: 0, max: 0.08 };
export const K_RANGE: Range = { min: 0.03, max: 0.07 };

export const DEFAULT_F = 0.037;
export const DEFAULT_K = 0.06;

export type ParamListener = (params: { F: number; k: number }) => void;

class ParamStore {
  F = DEFAULT_F;
  k = DEFAULT_K;
  private listeners = new Set<ParamListener>();

  setF(value: number): void {
    this.F = clamp(value, F_RANGE.min, F_RANGE.max);
    this.emit();
  }

  setK(value: number): void {
    this.k = clamp(value, K_RANGE.min, K_RANGE.max);
    this.emit();
  }

  subscribe(listener: ParamListener): () => void {
    this.listeners.add(listener);
    listener({ F: this.F, k: this.k });
    return () => this.listeners.delete(listener);
  }

  private emit(): void {
    for (const listener of this.listeners) listener({ F: this.F, k: this.k });
  }
}

function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}

export const paramStore = new ParamStore();
