# Final-Review Fixes — Conductor / วาทยากร

**Branch:** feature/conductor-ux-v2  
**Build:** ✓ tsc + vite build — 0 errors, 302.95 kB JS, 14.34 kB CSS

---

## Changes Applied

### 1. Phrases panel overlay gating — `frontend/src/App.tsx`

- **Line 188**: Wrapped `<div className="result-glass-panel...">` block in `{screen === "camera" && (...)}` so the panel is only mounted on the camera tab.
- **Line 199**: Changed `<BottomNav onChange={setScreen} />` → `onChange={(s) => { setScreen(s); setPhrasesOpen(false); }}` so switching tabs closes the panel.

### 2. Close-button aria-label — `frontend/src/i18n/th.ts`, `en.ts`, `App.tsx`

- **th.ts line 74**: Added `close: "ปิด"` under new `// Misc actions` comment.
- **en.ts line 70**: Added `close: "Close"` (matching position; typed `: typeof th` so build would have failed without this).
- **App.tsx line 191**: Changed close button `aria-label` from `th.navSettings` → `th.close`.

### 3. Relative time in History — `frontend/src/i18n/th.ts`, `en.ts`, `components/HistoryScreen.tsx`

- **th.ts lines 77–80**: Added `timeJustNow`, `timeMinutesAgo`, `timeHoursAgo`, `timeDaysAgo` keys.
- **en.ts lines 73–76**: Added matching English keys.
- **HistoryScreen.tsx lines 15–23**: Replaced `formatTime` helper with `formatRelative(ts, t, lang)` that uses i18n dict for sub-7-day strings and falls back to `toLocaleDateString` for older entries.
- **HistoryScreen.tsx line 63**: Updated call site to `formatRelative(it.ts, th, lang)`.

### 4. a11y polish

| File | Change |
|------|--------|
| `SettingsScreen.tsx` line 34 | `aria-pressed={lang === "th"}` on Thai button |
| `SettingsScreen.tsx` line 37 | `aria-pressed={lang === "en"}` on EN button |
| `SettingsScreen.tsx` lines 27, 45, 67 | `aria-hidden="true"` on all three leading decorative SVGs |
| `BottomNav.tsx` line 56 | `aria-hidden="true"` on `<span className="bottom-nav-ico">` wrapper |
| `HistoryScreen.tsx` line 50 | `aria-hidden="true"` on empty-state clock SVG |
| `HistoryScreen.tsx` line 65 | `aria-hidden="true"` on speaker SVG in each history row |
| `ResultCard.tsx` line 49 | `aria-hidden="true"` on speaker button's SVG |

### 5. CSS fix — `frontend/src/components/ResultCard.tsx`

- **Lines 63–64**: Fixed invalid CSS `-var(--space-5)` → `calc(-1 * var(--space-5))` for both `left` and `right` on the loading-bar `<div>`.

---

## Build Output

```
vite v5.4.21 building for production...
✓ 48 modules transformed.
dist/index.html                  0.57 kB │ gzip:  0.44 kB
dist/assets/index-D0BlAvDU.css  14.34 kB │ gzip:  3.39 kB
dist/assets/index-f1lB0fr7.js  302.95 kB │ gzip: 94.81 kB
✓ built in 1.39s
```

## Nothing Skipped

All 5 requested fix groups were applied. No items could not be completed.
