"use client";

import { useMemo } from "react";
import { ChatPanel } from "@/components/ChatPanel";
import { ResultsGrid } from "@/components/ResultsGrid";
import {
  MAX_SELECTED_FOR_FEEDBACK,
  useFashionSearch,
} from "@/hooks/useFashionSearch";

export default function Home() {
  const {
    messages,
    products,
    visibleProducts,
    hasMore,
    showMore,
    searchState,
    isLoading,
    selectedProducts,
    groundingMode,
    setGroundingMode,
    sendMessage,
    submitFeedback,
    clearChat,
    removePositiveConstraint,
    toggleSelect,
  } = useFashionSearch();

  const selectedIds = useMemo(
    () => new Set(selectedProducts.keys()),
    [selectedProducts]
  );
  const selectedList = useMemo(
    () => Array.from(selectedProducts.values()),
    [selectedProducts]
  );
  const selectionFull = selectedProducts.size >= MAX_SELECTED_FOR_FEEDBACK;

  return (
    <div className="flex h-screen overflow-hidden bg-white">
      {/* ── Left: Chat (input handles both messages and relevance feedback) ── */}
      <div className="w-[480px] flex-shrink-0 flex flex-col border-r border-gray-100">
        <ChatPanel
          messages={messages}
          isLoading={isLoading}
          groundingMode={groundingMode}
          onGroundingModeChange={setGroundingMode}
          onSend={sendMessage}
          onSubmitFeedback={submitFeedback}
          selected={selectedList}
          onRemoveSelected={toggleSelect}
          onClear={clearChat}
        />
      </div>

      {/* ── Right: Results ── */}
      <div className="flex-1 flex flex-col min-w-0">
        <ResultsGrid
          products={visibleProducts}
          totalProducts={products.length}
          hasMore={hasMore}
          onShowMore={showMore}
          searchState={searchState}
          isLoading={isLoading}
          selectedIds={selectedIds}
          selectionFull={selectionFull}
          onToggleSelect={toggleSelect}
          onRemoveConstraint={removePositiveConstraint}
        />
      </div>
    </div>
  );
}
