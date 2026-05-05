"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Check } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { Product, SearchState } from "@/types";

interface ResultsGridProps {
  products: Product[];
  totalProducts: number;
  hasMore: boolean;
  onShowMore: () => void;
  searchState: SearchState;
  isLoading: boolean;
  selectedIds: Set<string>;
  selectionFull: boolean;
  onToggleSelect: (product: Product) => void;
  onRemoveConstraint?: (constraint: string) => void;
}

export function ResultsGrid({
  products,
  totalProducts,
  hasMore,
  onShowMore,
  searchState,
  isLoading,
  selectedIds,
  selectionFull,
  onToggleSelect,
  onRemoveConstraint,
}: ResultsGridProps) {
  const hasQuery = !!searchState.current_query;
  const hasResults = products.length > 0;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-5 py-3 border-b border-gray-100 flex-shrink-0">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-900">Results</h2>
          {hasResults && (
            <span className="text-xs text-gray-400">
              {products.length === totalProducts
                ? `${totalProducts} items`
                : `${products.length} of ${totalProducts} items`}
            </span>
          )}
        </div>

        {/* Active search state summary */}
        {hasQuery && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            <Badge variant="outline" size="sm">
              <span className="text-gray-400 mr-0.5">query:</span>
              {searchState.current_query}
            </Badge>
            {searchState.positive_constraints.map((c) => (
              <Badge key={c} variant="positive" size="sm" className="group/badge">
                ✓ {c}
                {onRemoveConstraint && (
                  <button
                    onClick={() => onRemoveConstraint(c)}
                    className="ml-0.5 opacity-50 hover:opacity-100 transition-opacity leading-none"
                    aria-label={`Remove ${c}`}
                  >
                    ×
                  </button>
                )}
              </Badge>
            ))}
            {searchState.negative_constraints.map((c) => (
              <Badge key={c} variant="negative" size="sm">
                ✗ {c}
              </Badge>
            ))}
            {searchState.style_tags.map((t) => (
              <Badge key={t} variant="style" size="sm">
                {t}
              </Badge>
            ))}
            {searchState.occasion && (
              <Badge variant="secondary" size="sm">
                {searchState.occasion}
              </Badge>
            )}
          </div>
        )}

        {/* Selection helper */}
        {hasResults && (
          <p className="mt-2 text-[11px] text-gray-400">
            {selectedIds.size === 0
              ? "Tap up to 3 items to send relevance feedback."
              : `${selectedIds.size} of 3 selected${
                  selectionFull ? " — at the limit" : ""
                }.`}
          </p>
        )}
      </div>

      {/* Grid area */}
      <div className="flex-1 overflow-y-auto px-5 py-4">
        {!hasQuery && !isLoading ? (
          <EmptyState />
        ) : isLoading && !hasResults ? (
          <LoadingSkeleton />
        ) : !hasResults ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <p className="text-sm text-gray-500">No results found.</p>
            <p className="text-xs text-gray-400 mt-1">Try a broader query.</p>
          </div>
        ) : (
          <>
            <ProductGrid
              products={products}
              isLoading={isLoading}
              selectedIds={selectedIds}
              selectionFull={selectionFull}
              onToggleSelect={onToggleSelect}
            />
            {hasMore && (
              <div className="flex justify-center pt-6 pb-2">
                <button
                  onClick={onShowMore}
                  className="px-5 py-2 rounded-lg border border-gray-200 text-sm font-medium text-gray-600 hover:bg-gray-50 hover:border-gray-300 transition-colors"
                >
                  Show 200 more
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── Product grid ──────────────────────────────────────────────────────────────

function ProductGrid({
  products,
  isLoading,
  selectedIds,
  selectionFull,
  onToggleSelect,
}: {
  products: Product[];
  isLoading: boolean;
  selectedIds: Set<string>;
  selectionFull: boolean;
  onToggleSelect: (p: Product) => void;
}) {
  return (
    <div className="grid grid-cols-6 gap-3">
      <AnimatePresence mode="popLayout">
        {products.map((product, i) => (
          <motion.div
            key={product.id}
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ delay: Math.min(i * 0.04, 0.3), duration: 0.2 }}
            className={isLoading ? "opacity-50 pointer-events-none" : ""}
          >
            <ProductCard
              product={product}
              isSelected={selectedIds.has(product.id)}
              selectionFull={selectionFull}
              onToggleSelect={onToggleSelect}
            />
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

function ProductCard({
  product,
  isSelected,
  selectionFull,
  onToggleSelect,
}: {
  product: Product;
  isSelected: boolean;
  selectionFull: boolean;
  onToggleSelect: (p: Product) => void;
}) {
  const [imgError, setImgError] = useState(false);
  const disabled = !isSelected && selectionFull;

  return (
    <div
      className={[
        "group rounded-xl overflow-hidden border bg-white transition-all duration-150",
        disabled
          ? "border-gray-100 opacity-50 cursor-not-allowed"
          : "cursor-pointer",
        isSelected
          ? "border-rose-400 shadow-[0_0_0_2px_rgb(251,113,133)]"
          : !disabled
          ? "border-gray-100 hover:border-gray-200 hover:shadow-sm"
          : "",
      ].join(" ")}
      onClick={() => {
        if (disabled) return;
        onToggleSelect(product);
      }}
    >
      {/* Image */}
      <div className="aspect-[3/4] bg-gray-50 overflow-hidden relative">
        {!imgError ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={product.image_url}
            alt={product.category ?? "Fashion item"}
            onError={() => setImgError(true)}
            loading="lazy"
            className="w-full h-full object-cover group-hover:scale-[1.02] transition-transform duration-300"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-300 text-2xl">
            👗
          </div>
        )}

        {/* Selection badge */}
        {isSelected && (
          <div className="absolute top-1.5 right-1.5 h-5 w-5 rounded-full bg-rose-500 text-white flex items-center justify-center shadow">
            <Check className="h-3 w-3" />
          </div>
        )}
      </div>

      {/* Info */}
      <div className="px-2.5 py-2">
        {product.category && (
          <p className="text-xs font-medium text-gray-700 truncate capitalize">
            {product.category}
          </p>
        )}
        {product.score !== undefined && (
          <p className="text-[11px] text-gray-400 mt-0.5">
            {(product.score * 100).toFixed(1)}% match
          </p>
        )}
        {product.attributes && (
          <div className="flex flex-wrap gap-1 mt-1.5">
            {Object.values(product.attributes)
              .flat()
              .map((val) => (
                <span
                  key={val}
                  className="text-[10px] px-1.5 py-px rounded-full bg-gray-100 text-gray-500"
                >
                  {val}
                </span>
              ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Empty / loading states ────────────────────────────────────────────────────

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center">
      <div className="w-16 h-16 rounded-2xl bg-gray-50 border border-gray-100 flex items-center justify-center mb-4 text-3xl">
        👘
      </div>
      <p className="text-sm font-medium text-gray-700">
        Your results will appear here
      </p>
      <p className="text-xs text-gray-400 mt-1.5 max-w-[200px] leading-relaxed">
        Start a conversation on the left to search the fashion catalog.
      </p>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="grid grid-cols-6 gap-3">
      {Array.from({ length: 18 }).map((_, i) => (
        <div
          key={i}
          className="rounded-xl overflow-hidden border border-gray-100"
        >
          <div className="aspect-[3/4] bg-gray-100 animate-pulse" />
          <div className="px-2.5 py-2 space-y-1.5">
            <div className="h-3 w-2/3 bg-gray-100 rounded animate-pulse" />
            <div className="h-2.5 w-1/2 bg-gray-100 rounded animate-pulse" />
          </div>
        </div>
      ))}
    </div>
  );
}
