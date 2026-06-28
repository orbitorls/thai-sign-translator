import React, { useEffect, useMemo, useRef, useState } from "react";
import { useT } from "../i18n";
import { useSettings } from "../hooks/SettingsProvider";
import { MOCK_SIGNS_TH, MOCKUP_MODE } from "../mockup";

interface SignEntry {
  id: string;
  label_th: string;
  label_en: string;
  category: string;
  description?: string;
  description_th?: string;
  description_en?: string;
}

export function DictionaryScreen() {
  const { t, lang } = useT();
  const { settings } = useSettings();
  const [search, setSearch] = useState("");
  const [activeCategory, setActiveCategory] = useState("all");
  const [playingId, setPlayingId] = useState<string | null>(null);
  const playTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (playTimerRef.current) window.clearInterval(playTimerRef.current);
    };
  }, []);

  const signs = MOCK_SIGNS_TH as SignEntry[];

  const categories = useMemo(() => {
    const seen = new Set<string>();
    return signs
      .map((s) => s.category)
      .filter((c) => {
        if (seen.has(c)) return false;
        seen.add(c);
        return true;
      });
  }, [signs]);

  const filtered = signs.filter((s) => {
    const q = search.trim().toLowerCase();
    const matchCat = activeCategory === "all" || s.category === activeCategory;
    const label = lang === "th" ? s.label_th : s.label_en;
    const matchSearch =
      !q ||
      label.toLowerCase().includes(q) ||
      s.label_en.toLowerCase().includes(q) ||
      s.label_th.includes(q);
    return matchCat && matchSearch;
  });

  function playSign(id: string) {
    setPlayingId(id);
    if (playTimerRef.current) window.clearInterval(playTimerRef.current);
    playTimerRef.current = window.setInterval(() => setPlayingId((p) => (p === id ? id : p)), 400);
    window.setTimeout(() => {
      if (playTimerRef.current) window.clearInterval(playTimerRef.current);
      setPlayingId(null);
    }, 2000);
  }

  return (
    <div className="screen-sheet">
      <h2 className="sheet-title" style={{ marginBottom: "var(--space-4)" }}>
        {t.signLibrary}
      </h2>
      <p style={{ color: "rgba(255,255,255,0.6)", marginBottom: "var(--space-4)", fontSize: "var(--font-size-sm)" }}>
        {t.signLibraryDesc}
      </p>

      <input
        type="search"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder={t.searchDictionary}
        className="glass-input"
        style={{
          width: "100%",
          padding: "var(--space-3)",
          marginBottom: "var(--space-3)",
          borderRadius: "var(--radius-md)",
          border: "1px solid rgba(255,255,255,0.2)",
          background: "rgba(0,0,0,0.25)",
          color: "#fff",
          fontFamily: "var(--font-family)",
        }}
      />

      <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--space-2)", marginBottom: "var(--space-4)" }}>
        <button
          type="button"
          className={`glass-chip${activeCategory === "all" ? " on" : ""}`}
          onClick={() => setActiveCategory("all")}
        >
          {t.allCategories}
        </button>
        {categories.map((c) => (
          <button
            key={c}
            type="button"
            className={`glass-chip${activeCategory === c ? " on" : ""}`}
            onClick={() => setActiveCategory(c)}
          >
            {c}
          </button>
        ))}
      </div>

      {filtered.length === 0 ? (
        <div className="empty-state">
          <p>{t.dictionaryEmpty}</p>
          <p style={{ fontSize: "var(--font-size-sm)" }}>{t.dictionaryEmptyHint}</p>
        </div>
      ) : (
        <ul className="history-list">
          {filtered.map((sign) => {
            const label = lang === "th" ? sign.label_th : sign.label_en;
            return (
              <li key={sign.id}>
                <button type="button" className="history-row" onClick={() => playSign(sign.id)}>
                  <div className="history-word">
                    <b>{label}</b>
                    <small>{sign.category}</small>
                  </div>
                  <span style={{ fontSize: "var(--font-size-xs)", color: "rgba(255,255,255,0.7)" }}>
                    {playingId === sign.id ? "▶" : t.playAnimation}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}

      {MOCKUP_MODE && (
        <p style={{ marginTop: "var(--space-4)", fontSize: "var(--font-size-xs)", color: "rgba(255,255,255,0.45)" }}>
          {settings.diagnosticsEnabled ? "" : ""}
        </p>
      )}
    </div>
  );
}
