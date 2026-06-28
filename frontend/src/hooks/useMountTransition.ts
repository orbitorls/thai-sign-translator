import { useEffect, useRef, useState } from "react";

/**
 * Delays unmounting by `duration` ms so exit animations can play.
 * Returns { mounted, closing }:
 *   - mounted: whether the element should be in the DOM
 *   - closing: true during the exit animation window
 *
 * Respects prefers-reduced-motion — uses 0ms delay so DOM never lingers.
 */
export function useMountTransition(
  open: boolean,
  duration = 240
): { mounted: boolean; closing: boolean } {
  const [mounted, setMounted] = useState(open);
  const [closing, setClosing] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const reduced =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  useEffect(() => {
    if (open) {
      if (timerRef.current) clearTimeout(timerRef.current);
      setClosing(false);
      setMounted(true);
    } else if (mounted) {
      setClosing(true);
      const delay = reduced ? 0 : duration;
      timerRef.current = setTimeout(() => {
        setMounted(false);
        setClosing(false);
      }, delay);
    }
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  return { mounted, closing };
}
