import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite';

let detectorPromise: Promise<FaceDetector | null> | null = null;

function loadDetector(): Promise<FaceDetector | null> {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      try {
        const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
        return await FaceDetector.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
          runningMode: 'IMAGE',
          minDetectionConfidence: 0.5,
        });
      } catch (err) {
        console.warn('Face detector unavailable, face mask will stay empty:', err);
        return null;
      }
    })();
  }
  return detectorPromise;
}

export interface FaceBox {
  /** Normalized [0, 1] bounding box, origin top-left. */
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Detects faces in `source` (whose pixel size is sourceWidth x sourceHeight). */
export async function detectFaces(
  source: HTMLImageElement | HTMLCanvasElement,
  sourceWidth: number,
  sourceHeight: number,
): Promise<FaceBox[]> {
  const detector = await loadDetector();
  if (!detector) return [];
  try {
    const result = detector.detect(source);
    return result.detections
      .filter((d) => d.boundingBox)
      .map((d) => {
        const bb = d.boundingBox!;
        return {
          x: bb.originX / sourceWidth,
          y: bb.originY / sourceHeight,
          width: bb.width / sourceWidth,
          height: bb.height / sourceHeight,
        };
      });
  } catch (err) {
    console.warn('Face detection failed:', err);
    return [];
  }
}
