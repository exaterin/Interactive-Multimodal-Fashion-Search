"use client";

import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { SuggestionChips } from "@/components/SuggestionChips";
import type { Message } from "@/types";

interface MessageListProps {
  messages: Message[];
  isLoading: boolean;
  onSuggestionSelect: (text: string) => void;
}

export function MessageList({
  messages,
  isLoading,
  onSuggestionSelect,
}: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to newest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  if (!messages.length) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center px-8">
        <div className="w-12 h-12 rounded-2xl bg-gray-100 flex items-center justify-center mb-4 text-2xl">
          🔍
        </div>
        <p className="text-sm font-medium text-gray-900">
          What are you looking for?
        </p>
        <p className="text-xs text-gray-400 mt-1">
          Describe a fashion item and I&apos;ll find it for you.
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
      <AnimatePresence initial={false}>
        {messages.map((msg, idx) => {
          const isUser = msg.role === "user";
          const isLast = idx === messages.length - 1;

          return (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              className={`flex ${isUser ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] ${isUser ? "items-end" : "items-start"} flex flex-col gap-1`}
              >
                {/* Liked-item thumbnails (shown above the bubble for visual similarity requests) */}
                {isUser && msg.likedImages && msg.likedImages.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 justify-end mb-1">
                    {msg.likedImages.slice(0, 6).map((url, i) => (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        key={i}
                        src={url}
                        alt="liked item"
                        className="h-14 w-10 object-cover rounded-lg border border-white/20"
                      />
                    ))}
                    {msg.likedImages.length > 6 && (
                      <div className="h-14 w-10 rounded-lg bg-gray-700 flex items-center justify-center text-white text-xs font-medium">
                        +{msg.likedImages.length - 6}
                      </div>
                    )}
                  </div>
                )}

                {/* Bubble */}
                <div
                  className={`px-3.5 py-2.5 rounded-2xl text-sm leading-relaxed ${
                    isUser
                      ? "bg-gray-900 text-white rounded-br-sm"
                      : "bg-gray-50 text-gray-900 border border-gray-100 rounded-bl-sm"
                  }`}
                >
                  {msg.content}
                </div>

                {/* Suggestion chips — only on the latest assistant message */}
                {!isUser && isLast && msg.suggestions?.length ? (
                  <SuggestionChips
                    suggestions={msg.suggestions}
                    onSelect={onSuggestionSelect}
                    disabled={isLoading}
                  />
                ) : null}
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>

      {/* Typing indicator */}
      {isLoading && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex justify-start"
        >
          <div className="px-4 py-3 rounded-2xl rounded-bl-sm bg-gray-50 border border-gray-100">
            <TypingDots />
          </div>
        </motion.div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}

function TypingDots() {
  return (
    <div className="flex gap-1 items-center h-4">
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-gray-400"
          animate={{ y: [0, -3, 0] }}
          transition={{
            duration: 0.6,
            repeat: Infinity,
            delay: i * 0.15,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}
