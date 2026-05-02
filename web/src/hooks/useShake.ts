import { useCallback, useEffect, useRef, useState } from "react";

export type Jitter = readonly [number, number];

const ZERO_JITTER: Jitter = [0, 0];

function zeroJitterArray(count: number): Jitter[] {
  return Array.from({ length: count }, () => ZERO_JITTER);
}

interface ShakeOptions {
  count?: number;
  durationMs?: number;
  maxAmplitudePx?: number;
}

// Position-level jitter for the board. Triggering returns a fresh random
// offset per node every animation frame, with quadratic ease-out so the shake
// fades naturally. Re-triggering while active resets the timer for a longer
// shudder; the rAF loop is recycled.
export function useShake({
  count = 24,
  durationMs = 1400,
  maxAmplitudePx = 4,
}: ShakeOptions = {}) {
  const [jitter, setJitter] = useState<Jitter[]>(() => zeroJitterArray(count));
  const rafRef = useRef<number | null>(null);
  const startRef = useRef(0);

  const stop = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setJitter(zeroJitterArray(count));
  }, [count]);

  const trigger = useCallback(() => {
    startRef.current = performance.now();
    if (rafRef.current !== null) return; // existing loop will read the new start
    const tick = (now: number) => {
      const elapsed = now - startRef.current;
      if (elapsed >= durationMs) {
        rafRef.current = null;
        setJitter(zeroJitterArray(count));
        return;
      }
      const decay = 1 - elapsed / durationMs;
      const amplitude = maxAmplitudePx * decay * decay;
      const next: Jitter[] = Array.from({ length: count }, () => [
        (Math.random() - 0.5) * 2 * amplitude,
        (Math.random() - 0.5) * 2 * amplitude,
      ]);
      setJitter(next);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [count, durationMs, maxAmplitudePx]);

  useEffect(
    () => () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    },
    [],
  );

  return { jitter, trigger, stop };
}
