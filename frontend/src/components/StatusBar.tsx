import React from "react";

interface StatusBarProps {
  message: string;
  type?: "info" | "warn" | "error" | "success";
}

const colors = {
  info: "var(--color-text-muted)",
  warn: "var(--color-warn)",
  error: "var(--color-danger)",
  success: "var(--color-success)",
};

export function StatusBar({ message, type = "info" }: StatusBarProps) {
  if (!message) return null;
  return (
    <p
      role="status"
      style={{
        fontSize: "var(--font-size-sm)",
        color: colors[type],
        textAlign: "center",
        minHeight: "1.5em",
        fontWeight: 600,
      }}
    >
      {message}
    </p>
  );
}
