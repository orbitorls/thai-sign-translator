# Final Fix Report — 2026-06-20

## Fixes Made

### Finding 1 — Empty-frames path validates model id (app.py:357-360)
**File:** `src/tsl/api/app.py`, lines 357–360 (translate() function)

Old code echoed `req.model` verbatim when `req.frames` was empty, letting garbage model ids
return 200. Replaced with the same `get_spec()` / `HTTPException(400)` pattern used by the
non-empty path:
```python
if not req.frames:
    spec = get_spec(req.model) if req.model is not None else default_spec()
    if spec is None:
        raise HTTPException(status_code=400, detail=f"unknown model {req.model!r}")
    return TranslateResponse(sentence="", score=0.0, model=spec.id)
```

### Finding 2 — Dispatch assertions added to two tests (test_translate_model_param.py)
**File:** `tests/api/test_translate_model_param.py`

- `test_translate_omitted_model_uses_default`: added `mock_fn.assert_called_once_with(None)` after the `with` block (line ~49)
- `test_translate_explicit_model_v2_dispatches_correctly`: added `mock_fn.assert_called_once_with("v2_slt")` after the `with` block (line ~65)

### Finding 3 — Build comment added near _FRONTEND_DIST (app.py:64-65)
**File:** `src/tsl/api/app.py`, lines 64–65

Added two-line comment explaining the Vite build step that must be run before serving
in production.

## Test Results

```
44 passed, 2 warnings in 22.73s
```

All 44 tests in `tests/api/` pass. No regressions.

## TypeScript Check

`cd frontend && npx tsc --noEmit` — exit 0, clean.

## Concerns

None. All three findings are addressed, tests green, TS clean.
