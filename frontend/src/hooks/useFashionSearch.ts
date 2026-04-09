"use client";

import { useState, useCallback } from "react";
import { sendChatMessage, resetChat } from "@/lib/api";
import { makeId, initialSearchState } from "@/lib/utils";
import type { Message, Product, SearchState, LikedItem } from "@/types";

interface UseFashionSearch {
  messages: Message[];
  products: Product[];
  searchState: SearchState;
  isLoading: boolean;
  error: string | null;
  likedProducts: Map<string, Product>;
  sendMessage: (text: string) => Promise<void>;
  clearChat: () => void;
  removePositiveConstraint: (constraint: string) => Promise<void>;
  toggleLike: (product: Product) => void;
  clearLiked: () => void;
}

export function useFashionSearch(): UseFashionSearch {
  const [messages, setMessages] = useState<Message[]>([]);
  const [products, setProducts] = useState<Product[]>([]);
  const [searchState, setSearchState] = useState<SearchState>(
    initialSearchState()
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [likedProducts, setLikedProducts] = useState<Map<string, Product>>(
    new Map()
  );

  const toggleLike = useCallback((product: Product) => {
    setLikedProducts((prev) => {
      const next = new Map(prev);
      if (next.has(product.id)) {
        next.delete(product.id);
      } else {
        next.set(product.id, product);
      }
      return next;
    });
  }, []);

  const clearLiked = useCallback(() => {
    setLikedProducts(new Map());
  }, []);

  const _buildLikedItems = (map: Map<string, Product>): LikedItem[] =>
    Array.from(map.values()).map((p) => ({
      id: p.id,
      category: p.category,
      attributes: p.attributes,
    }));

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
          liked_items: _buildLikedItems(likedProducts),
        });

        const assistantMsg: Message = {
          id: makeId(),
          role: "assistant",
          content: response.message,
          suggestions: response.suggestions ?? [],
          timestamp: new Date(),
        };

        setMessages((prev) => [...prev, assistantMsg]);
        setProducts(response.products ?? []);
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
    [isLoading, searchState, likedProducts]
  );

  const clearChat = useCallback(async () => {
    resetChat().catch(() => null);
    setMessages([]);
    setProducts([]);
    setSearchState(initialSearchState());
    setError(null);
    setLikedProducts(new Map());
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
          liked_items: _buildLikedItems(likedProducts),
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
    [isLoading, searchState, likedProducts]
  );

  return {
    messages,
    products,
    searchState,
    isLoading,
    error,
    likedProducts,
    sendMessage,
    clearChat,
    removePositiveConstraint,
    toggleLike,
    clearLiked,
  };
}
