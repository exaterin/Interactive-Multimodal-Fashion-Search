"use client";

import { useState, useCallback } from "react";
import { sendChatMessage, sendRelevanceFeedback, resetChat } from "@/lib/api";
import { makeId, initialSearchState } from "@/lib/utils";
import type { Message, Product, SearchState, LikedItem, HistoryMessage } from "@/types";

export const MAX_SELECTED_FOR_FEEDBACK = 3;

interface UseFashionSearch {
  messages: Message[];
  products: Product[];
  visibleProducts: Product[];
  hasMore: boolean;
  showMore: () => void;
  searchState: SearchState;
  isLoading: boolean;
  error: string | null;
  selectedProducts: Map<string, Product>;
  groundingMode: "attribute" | "description" | "image";
  setGroundingMode: (mode: "attribute" | "description" | "image") => void;
  sendMessage: (text: string) => Promise<void>;
  submitFeedback: (comment: string) => Promise<void>;
  clearChat: () => void;
  removePositiveConstraint: (constraint: string) => Promise<void>;
  toggleSelect: (product: Product) => void;
  clearSelection: () => void;
}

export function useFashionSearch(): UseFashionSearch {
  const [messages, setMessages] = useState<Message[]>([]);
  const [products, setProducts] = useState<Product[]>([]);
  const [visibleCount, setVisibleCount] = useState(200);
  const [searchState, setSearchState] = useState<SearchState>(
    initialSearchState()
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedProducts, setSelectedProducts] = useState<Map<string, Product>>(
    new Map()
  );
  const [groundingMode, setGroundingMode] = useState<"attribute" | "description" | "image">(
    "attribute"
  );

  const visibleProducts = products.slice(0, visibleCount);
  const hasMore = visibleCount < products.length;

  const showMore = useCallback(() => {
    setVisibleCount((prev) => prev + 200);
  }, []);

  const toggleSelect = useCallback((product: Product) => {
    setSelectedProducts((prev) => {
      const next = new Map(prev);
      if (next.has(product.id)) {
        next.delete(product.id);
        return next;
      }
      if (next.size >= MAX_SELECTED_FOR_FEEDBACK) {
        return prev;
      }
      next.set(product.id, product);
      return next;
    });
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedProducts(new Map());
  }, []);

  const _toLikedItems = (map: Map<string, Product>): LikedItem[] =>
    Array.from(map.values()).map((p) => ({
      id: p.id,
      category: p.category,
      attributes: p.attributes,
    }));

  const _buildHistory = (msgs: Message[]): HistoryMessage[] =>
    msgs.map((m) => ({ role: m.role, content: m.content }));

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isLoading) return;
      setError(null);

      const userMsg: Message = {
        id: makeId(),
        role: "user",
        content: text.trim(),
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);

      try {
        const response = await sendChatMessage({
          message: text.trim(),
          search_state: searchState,
          liked_items: [], // relevance feedback now flows through /feedback
          grounding_mode: groundingMode,
          chat_history: _buildHistory(messages),
        });

        if (response.intent === "reset") {
          setMessages([]);
          setProducts([]);
          setVisibleCount(200);
          setSearchState(initialSearchState());
          setSelectedProducts(new Map());
          return;
        }

        const assistantMsg: Message = {
          id: makeId(),
          role: "assistant",
          content: response.message,
          suggestions: response.suggestions ?? [],
          timestamp: new Date(),
        };

        const clearsHistory =
          response.intent === "initial_search" || response.intent === "new_query";
        setMessages(clearsHistory ? [userMsg, assistantMsg] : (prev) => [...prev, assistantMsg]);
        setProducts(response.products ?? []);
        setVisibleCount(200);
        setSearchState(response.search_state ?? searchState);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unexpected error";
        setError(message);

        setMessages((prev) => [
          ...prev,
          {
            id: makeId(),
            role: "assistant",
            content: `Sorry, something went wrong: ${message}`,
            timestamp: new Date(),
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading, searchState, groundingMode, messages]
  );

  const submitFeedback = useCallback(
    async (comment: string) => {
      if (isLoading) return;
      if (selectedProducts.size === 0 || selectedProducts.size > MAX_SELECTED_FOR_FEEDBACK) return;

      setError(null);

      const selectedList = Array.from(selectedProducts.values());
      const trimmedComment = comment.trim();

      const userContent =
        trimmedComment ||
        `Use these ${selectedList.length} item${
          selectedList.length !== 1 ? "s" : ""
        } as a relevance signal`;

      const userMsg: Message = {
        id: makeId(),
        role: "user",
        content: userContent,
        likedImages: selectedList.map((p) => p.image_url),
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);

      try {
        const response = await sendRelevanceFeedback({
          selected_items: _toLikedItems(selectedProducts),
          comment: trimmedComment,
          search_state: searchState,
          grounding_mode: groundingMode,
          chat_history: _buildHistory(messages),
        });

        setMessages((prev) => [
          ...prev,
          {
            id: makeId(),
            role: "assistant",
            content: response.message,
            suggestions: response.suggestions ?? [],
            timestamp: new Date(),
          },
        ]);
        setProducts(response.products ?? []);
        setVisibleCount(200);
        setSearchState(response.search_state ?? searchState);
        setSelectedProducts(new Map());
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unexpected error";
        setError(message);
        setMessages((prev) => [
          ...prev,
          {
            id: makeId(),
            role: "assistant",
            content: `Sorry, something went wrong: ${message}`,
            timestamp: new Date(),
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading, searchState, selectedProducts, groundingMode, messages]
  );

  const clearChat = useCallback(async () => {
    resetChat().catch(() => null);
    setMessages([]);
    setProducts([]);
    setVisibleCount(200);
    setSearchState(initialSearchState());
    setError(null);
    setSelectedProducts(new Map());
  }, []);

  const removePositiveConstraint = useCallback(
    async (constraint: string) => {
      if (isLoading) return;

      const updatedState: SearchState = {
        ...searchState,
        positive_constraints: searchState.positive_constraints.filter(
          (c) => c !== constraint
        ),
      };
      setSearchState(updatedState);
      setIsLoading(true);
      setError(null);

      const userMsg: Message = {
        id: makeId(),
        role: "user",
        content: `Remove "${constraint}" constraint`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMsg]);

      try {
        const response = await sendChatMessage({
          message: `The user removed the constraint "${constraint}". Refresh results without it.`,
          search_state: updatedState,
          liked_items: [],
          grounding_mode: groundingMode,
          chat_history: _buildHistory(messages),
        });

        setMessages((prev) => [
          ...prev,
          {
            id: makeId(),
            role: "assistant",
            content: response.message,
            suggestions: response.suggestions ?? [],
            timestamp: new Date(),
          },
        ]);
        setProducts(response.products ?? []);
        setVisibleCount(200);
        setSearchState(response.search_state ?? updatedState);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unexpected error";
        setError(message);
        setMessages((prev) => [
          ...prev,
          {
            id: makeId(),
            role: "assistant",
            content: `Sorry, something went wrong: ${message}`,
            timestamp: new Date(),
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading, searchState, groundingMode, messages]
  );

  return {
    messages,
    products,
    visibleProducts,
    hasMore,
    showMore,
    searchState,
    isLoading,
    error,
    selectedProducts,
    groundingMode,
    setGroundingMode,
    sendMessage,
    submitFeedback,
    clearChat,
    removePositiveConstraint,
    toggleSelect,
    clearSelection,
  };
}
