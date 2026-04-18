"use client";

import { ChatPanel } from "@/components/ChatPanel";
import { ResultsGrid } from "@/components/ResultsGrid";
import { useFashionSearch } from "@/hooks/useFashionSearch";

export default function Home() {
  const {
    messages,
    products,
    searchState,
    isLoading,
    likedProducts,
    sendMessage,
    findSimilar,
    clearChat,
    removePositiveConstraint,
    toggleLike,
  } = useFashionSearch();

  const likedId = likedProducts.size > 0 ? likedProducts.keys().next().value as string : null;

  return (
    <div className="flex h-screen overflow-hidden bg-white">
      {/* ── Left: Chat ── */}
      <div className="w-[480px] flex-shrink-0 flex flex-col border-r border-gray-100">
        <ChatPanel
          messages={messages}
          isLoading={isLoading}
          onSend={sendMessage}
          onClear={clearChat}
        />
      </div>

      {/* ── Right: Results ── */}
      <div className="flex-1 flex flex-col min-w-0">
        <ResultsGrid
          products={products}
          searchState={searchState}
          isLoading={isLoading}
          likedId={likedId}
          onToggleLike={toggleLike}
          onFindSimilar={findSimilar}
          onRemoveConstraint={removePositiveConstraint}
        />
      </div>
    </div>
  );
}
