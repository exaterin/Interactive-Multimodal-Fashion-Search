"use client";

import { Heart, RotateCcw, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { MessageList } from "@/components/MessageList";
import { InputBox } from "@/components/InputBox";
import type { Message } from "@/types";

interface ChatPanelProps {
  messages: Message[];
  isLoading: boolean;
  likedCount: number;
  onSend: (text: string) => void;
  onClear: () => void;
  onClearLiked: () => void;
}

export function ChatPanel({
  messages,
  isLoading,
  likedCount,
  onSend,
  onClear,
  onClearLiked,
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

      {/* Liked items banner */}
      {likedCount > 0 && (
        <div className="mx-4 mt-3 flex items-center justify-between rounded-xl bg-rose-50 border border-rose-100 px-3 py-2">
          <div className="flex items-center gap-2">
            <Heart className="h-3.5 w-3.5 text-rose-400 fill-rose-400 flex-shrink-0" />
            <span className="text-xs text-rose-700 font-medium">
              {likedCount} item{likedCount !== 1 ? "s" : ""} liked
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              disabled={isLoading}
              onClick={() => onSend("Show me more items like the ones I liked")}
              className="text-[11px] text-rose-600 hover:text-rose-800 font-medium disabled:opacity-40 transition-colors"
            >
              Find similar
            </button>
            <span className="text-rose-200">·</span>
            <button
              onClick={onClearLiked}
              className="text-gray-400 hover:text-gray-600 transition-colors"
              aria-label="Clear liked items"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        </div>
      )}

      {/* Messages */}
      <MessageList
        messages={messages}
        isLoading={isLoading}
        onSuggestionSelect={onSend}
      />

      {/* Input */}
      <div className="border-t border-gray-100">
        <InputBox
          onSend={onSend}
          disabled={isLoading}
          likedCount={likedCount}
        />
      </div>
    </div>
  );
}
