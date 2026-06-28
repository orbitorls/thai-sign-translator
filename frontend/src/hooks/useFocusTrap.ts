import { useEffect, useRef } from "react";

const FOCUSABLE_SELECTOR =
  'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * Traps keyboard focus inside `containerRef` while `active` is true.
 * Restores focus to `document.activeElement` at activation time on deactivation.
 * Also handles Escape key — calls `onEscape` if provided.
 */
export function useFocusTrap(
  containerRef: React.RefObject<HTMLElement | null>,
  active: boolean,
  onEscape?: () => void,
  initialFocus: "first" | "container" = "first"
): void {
  const savedFocusRef = useRef<Element | null>(null);

  useEffect(() => {
    if (!active) return;
    const container = containerRef.current;
    if (!container) return;

    // Save current focus
    savedFocusRef.current = document.activeElement;

    // Move focus into dialog
    if (initialFocus === "first") {
      const first = container.querySelector<HTMLElement>(FOCUSABLE_SELECTOR);
      first ? first.focus() : container.focus();
    } else {
      container.focus();
    }

    function handleKey(e: KeyboardEvent) {
      if (!container) return;
      if (e.key === "Escape" && onEscape) {
        e.preventDefault();
        onEscape();
        return;
      }
      if (e.key === "Tab") {
        const els = Array.from(
          container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)
        );
        if (els.length === 0) return;
        const first = els[0];
        const last = els[els.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    }

    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("keydown", handleKey);
      // Restore focus
      if (savedFocusRef.current && "focus" in savedFocusRef.current) {
        (savedFocusRef.current as HTMLElement).focus();
      }
    };
  }, [active, onEscape, initialFocus]);
}
