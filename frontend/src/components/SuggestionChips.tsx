"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";

interface SuggestionChipsProps {
  suggestions: string[];
  onSelect: (text: string) => void;
  disabled?: boolean;
}

export function SuggestionChips({
  suggestions,
  onSelect,
  disabled = false,
}: SuggestionChipsProps) {
  if (!suggestions.length) return null;

  return (
    <div className="flex flex-wrap gap-2 mt-2">
      {suggestions.map((s, i) => (
        <motion.div
          key={s}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.06, duration: 0.2 }}
        >
          <Button
            variant="chip"
            disabled={disabled}
            onClick={() => onSelect(s)}
          >
            {s}
          </Button>
        </motion.div>
      ))}
    </div>
  );
}
