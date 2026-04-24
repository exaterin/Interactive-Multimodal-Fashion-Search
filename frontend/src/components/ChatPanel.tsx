"use client";

import { RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { MessageList } from "@/components/MessageList";
import { InputBox } from "@/components/InputBox";
import type { Message } from "@/types";

interface ChatPanelProps {
  messages: Message[];
  isLoading: boolean;
  groundingMode: "attribute" | "description" | "image";
  onGroundingModeChange: (mode: "attribute" | "description" | "image") => void;
  onSend: (text: string) => void;
  onClear: () => void;
}

export function ChatPanel({
  messages,
  isLoading,
  groundingMode,
  onGroundingModeChange,
  onSend,
  onClear,
}: ChatPanelProps) {
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
        <div>
          <h1 className="text-sm font-semibold text-gray-900">Fashion Search</h1>
          <p className="text-[11px] text-gray-400 mt-0.5">
            Powered by FashionCLIP + LLM
          </p>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onClear}
          disabled={isLoading || !messages.length}
          className="text-gray-400 hover:text-gray-600 gap-1.5"
          aria-label="Clear chat"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          <span className="text-xs">Clear</span>
        </Button>
      </div>

      {/* Grounding mode toggle */}
      <div className="flex items-center gap-1 px-4 py-2 border-b border-gray-100 bg-gray-50">
        <span className="text-[11px] text-gray-400 mr-1">Grounding:</span>
        {(
          [
            { mode: "attribute", label: "Attribute-based" },
            { mode: "description", label: "Description-based" },
            { mode: "image", label: "Image-based" },
          ] as const
        ).map(({ mode, label }) => (
          <button
            key={mode}
            onClick={() => onGroundingModeChange(mode)}
            disabled={isLoading}
            className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
              groundingMode === mode
                ? "bg-gray-900 text-white"
                : "bg-white text-gray-500 border border-gray-200 hover:border-gray-400 hover:text-gray-700"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Messages */}
      <MessageList
        messages={messages}
        isLoading={isLoading}
        onSuggestionSelect={onSend}
      />

      {/* Input */}
      <div className="border-t border-gray-100">
        <InputBox onSend={onSend} disabled={isLoading} />
      </div>
    </div>
  );
}
