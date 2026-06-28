import { useEffect } from "react";
import { useLocation } from "react-router-dom";

const DEBUG_ENDPOINT =
  "http://127.0.0.1:7389/ingest/c1bf1fd9-7f85-4723-8e23-f87f6e861967";

function logLayout(hypothesisId: string, message: string, data: Record<string, unknown>) {
  // #region agent log
  fetch(DEBUG_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Debug-Session-Id": "5d0002",
    },
    body: JSON.stringify({
      sessionId: "5d0002",
      hypothesisId,
      location: "useLayoutDebug.ts",
      message,
      data,
      timestamp: Date.now(),
      runId: "mobile-layout",
    }),
  }).catch(() => {});
  // #endregion
}

/** Debug-only: logs mobile layout metrics (nav position, main width, overflow). */
export function useLayoutDebug() {
  const location = useLocation();

  useEffect(() => {
    function measure() {
      const main = document.querySelector("main");
      const navs = [...document.querySelectorAll("nav")];
      const mainRect = main?.getBoundingClientRect();
      const navMetrics = navs.map((n, i) => {
        const style = getComputedStyle(n);
        const rect = n.getBoundingClientRect();
        return {
          i,
          position: style.position,
          display: style.display,
          width: Math.round(rect.width),
          height: Math.round(rect.height),
          x: Math.round(rect.x),
          y: Math.round(rect.y),
        };
      });

      logLayout("A", "layout-metrics", {
        path: location.pathname,
        innerWidth: window.innerWidth,
        innerHeight: window.innerHeight,
        docScrollWidth: document.documentElement.scrollWidth,
        hOverflow: document.documentElement.scrollWidth > window.innerWidth + 1,
        mainWidth: mainRect ? Math.round(mainRect.width) : null,
        mainX: mainRect ? Math.round(mainRect.x) : null,
        navMetrics,
      });
    }

    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [location.pathname]);
}
