"use client";

import { useState, useRef, useCallback, KeyboardEvent } from "react";
import { motion } from "framer-motion";
import { Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface InputBoxProps {
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function InputBox({
  onSend,
  disabled = false,
  placeholder = "Describe what you're looking for…",
}: InputBoxProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [value, disabled, onSend]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    // Auto-grow up to ~4 lines
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 112)}px`;
  };

  return (
    <div className="px-4 pb-4 pt-2">
      <motion.div
        initial={false}
        className={cn(
          "flex items-end gap-2 rounded-xl border border-gray-200 bg-white px-3 py-2.5 shadow-sm",
          "focus-within:border-gray-400 focus-within:ring-2 focus-within:ring-gray-100",
          "transition-all duration-150"
        )}
      >
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder={placeholder}
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
          variant={value.trim() ? "default" : "ghost"}
          disabled={!value.trim() || disabled}
          onClick={handleSend}
          className="h-7 w-7 flex-shrink-0 rounded-lg transition-all"
          aria-label="Send message"
        >
          <Send className="h-3.5 w-3.5" />
        </Button>
      </motion.div>
      <p className="mt-1.5 text-center text-[11px] text-gray-300">
        Enter to send · Shift+Enter for new line
      </p>
    </div>
  );
}
