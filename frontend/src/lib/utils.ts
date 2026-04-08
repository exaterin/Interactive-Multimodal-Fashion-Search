import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function makeId(): string {
  return Math.random().toString(36).slice(2, 10);
}

export function initialSearchState() {
  return {
    original_query: "",
    current_query: "",
    positive_constraints: [] as string[],
    negative_constraints: [] as string[],
    style_tags: [] as string[],
    occasion: "",
  };
}
