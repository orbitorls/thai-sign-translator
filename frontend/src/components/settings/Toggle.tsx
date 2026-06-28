import React from "react";

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  id?: string;
  disabled?: boolean;
  "aria-label"?: string;
}

export const Toggle = React.memo(function Toggle({
  checked,
  onChange,
  id,
  disabled = false,
  "aria-label": ariaLabel,
}: ToggleProps) {
  return (
    <label
      htmlFor={id}
      className={`relative inline-flex items-center shrink-0 transition-opacity duration-200 ease-out ${
        disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer hover:brightness-105"
      }`}
    >
      <input
        id={id}
        type="checkbox"
        role="switch"
        aria-checked={checked}
        aria-label={ariaLabel}
        className="sr-only peer disabled:cursor-not-allowed"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <div
        className="glass-input w-14 h-7 rounded-full peer-checked:bg-primary/40 peer-focus-visible:ring-2 peer-focus-visible:ring-primary/40 peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-6 after:w-6 after:transition-all after:shadow-sm peer-disabled:opacity-50"
      />
    </label>
  );
});
