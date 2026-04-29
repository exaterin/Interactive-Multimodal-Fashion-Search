"use client";

import { useState, useRef, useCallback, KeyboardEvent } from "react";
import { motion } from "framer-motion";
import { Send, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Product } from "@/types";

interface InputBoxProps {
  onSend: (text: string) => void;
  onSubmitFeedback: (comment: string) => void;
  selected: Product[];
  onRemoveSelected: (product: Product) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function InputBox({
  onSend,
  onSubmitFeedback,
  selected,
  onRemoveSelected,
  disabled = false,
  placeholder,
}: InputBoxProps) {
  const inFeedbackMode = selected.length > 0;

  const resolvedPlaceholder =
    placeholder ??
    (inFeedbackMode
      ? `${selected.length} selected · add an optional comment, e.g. “like the color, but more elegant”…`
      : "Describe what you're looking for…");

  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (disabled) return;

    if (inFeedbackMode) {
      // Comment is optional in feedback mode.
      onSubmitFeedback(trimmed);
    } else {
      if (!trimmed) return;
      onSend(trimmed);
    }

    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [value, disabled, inFeedbackMode, onSend, onSubmitFeedback]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 112)}px`;
  };

  // In feedback mode the Send button is enabled even with no comment.
  const canSend = !disabled && (inFeedbackMode || !!value.trim());

  return (
    <div className="px-4 pb-4 pt-2">
      <motion.div
        initial={false}
        className={cn(
          "rounded-xl border bg-white shadow-sm transition-all duration-150",
          "focus-within:ring-2 focus-within:ring-gray-100",
          inFeedbackMode
            ? "border-rose-300 focus-within:border-rose-400"
            : "border-gray-200 focus-within:border-gray-400"
        )}
      >
        {/* Selected thumbnails (visible only in feedback mode) */}
        {inFeedbackMode && (
          <div className="flex items-center gap-1.5 px-3 pt-2.5">
            {selected.map((p) => (
              <div
                key={p.id}
                className="relative h-11 w-8 rounded-md overflow-hidden border border-gray-200 bg-gray-50 group"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={p.image_url}
                  alt={p.category ?? "selected item"}
                  className="h-full w-full object-cover"
                />
                <button
                  onClick={() => onRemoveSelected(p)}
                  disabled={disabled}
                  aria-label="Remove from selection"
                  className="absolute top-0.5 right-0.5 h-3.5 w-3.5 rounded-full bg-black/65 hover:bg-black/85 text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <X className="h-2 w-2" />
                </button>
              </div>
            ))}
            <span className="ml-1 text-[10px] text-rose-500 font-medium">
              relevance feedback
            </span>
          </div>
        )}

        <div className="flex items-end gap-2 px-3 py-2.5">
          <textarea
            ref={textareaRef}
            rows={1}
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder={resolvedPlaceholder}
            className={cn(
              "flex-1 resize-none bg-transparent text-sm text-gray-900",
              "placeholder:text-gray-400 focus:outline-none leading-relaxed",
              "disabled:cursor-not-allowed disabled:opacity-50",
              "max-h-28 overflow-y-auto"
            )}
            style={{ minHeight: "24px" }}
          />
          <Button
            size="icon"
            variant={canSend ? "default" : "ghost"}
            disabled={!canSend}
            onClick={handleSend}
            className="h-7 w-7 flex-shrink-0 rounded-lg transition-all"
            aria-label={inFeedbackMode ? "Send feedback" : "Send message"}
          >
            <Send className="h-3.5 w-3.5" />
          </Button>
        </div>
      </motion.div>
      <p className="mt-1.5 text-center text-[11px] text-gray-300">
        {inFeedbackMode
          ? "Enter to send feedback · comment is optional"
          : "Enter to send · Shift+Enter for new line"}
      </p>
    </div>
  );
}
