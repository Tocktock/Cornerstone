const defaultTheme = require('tailwindcss/defaultTheme');

module.exports = {
  content: [
    "./templates/**/*.html",
    "./templates/**/*.jinja",
    "./templates/**/*.j2",
    "./src/**/*.py",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", ...defaultTheme.fontFamily.sans],
      },
      colors: {
        primary: {
          DEFAULT: "#2563eb",
          foreground: "#ffffff",
        },
        slate: {
          950: "#0b1220",
        },
      },
      boxShadow: {
        card: "0 18px 38px rgba(15, 23, 42, 0.12)",
        button: "0 12px 28px rgba(37, 99, 235, 0.20)",
      },
    },
  },
  plugins: [],
};
