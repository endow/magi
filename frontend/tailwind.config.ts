import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: "#05080d",
          panel: "#0a1119",
          border: "#203040",
          text: "#d6e1ee",
          dim: "#8ca0b3",
          ok: "#74f58a",
          err: "#ff7070",
          accent: "#4fc3f7"
        }
      }
    }
  },
  plugins: []
};

export default config;
