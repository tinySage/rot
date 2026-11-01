let devicePromise: Promise<GPUDevice | null> | null = null;

/** Feature-detects WebGPU and requests a device, once. Never throws. */
export function requestWebGPUDevice(): Promise<GPUDevice | null> {
  if (!devicePromise) {
    devicePromise = (async () => {
      try {
        if (!('gpu' in navigator) || !navigator.gpu) return null;
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return null;
        const device = await adapter.requestDevice();
        device.lost.then((info) => {
          console.warn('WebGPU device lost:', info.message);
        });
        return device;
      } catch (err) {
        console.warn('WebGPU unavailable, falling back to WebGL2:', err);
        return null;
      }
    })();
  }
  return devicePromise;
}
