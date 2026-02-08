import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: "#04070c",
          panel: "#09111a",
          border: "#1a3248",
          text: "#d5e7f7",
          dim: "#89a4bc",
          ok: "#72f0a0",
          err: "#ff6f7f",
          accent: "#ff8a5c"
        }
      }
    }
  },
  plugins: []
};

export default config;
