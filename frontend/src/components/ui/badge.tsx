import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-ring",
  {
    variants: {
      variant: {
        default: "border-transparent bg-gray-900 text-white",
        secondary: "border-transparent bg-gray-100 text-gray-700",
        positive:
          "border-transparent bg-emerald-50 text-emerald-700 border-emerald-200",
        negative: "border-transparent bg-red-50 text-red-700 border-red-200",
        style: "border-transparent bg-violet-50 text-violet-700 border-violet-200",
        outline: "border-gray-200 text-gray-600",
      },
      size: {
        default: "px-2.5 py-0.5",
        sm: "px-2 py-px text-[11px]",
      },
    },
    defaultVariants: {
      variant: "secondary",
      size: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, size, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant, size }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
