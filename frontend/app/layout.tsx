import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MAGI v0",
  description: "Send one prompt to three LLMs in parallel"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
