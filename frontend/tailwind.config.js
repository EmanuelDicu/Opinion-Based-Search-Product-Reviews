/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'obs-background': '#000000',     // pure black background
        'obs-panel': '#111111',          // dark panel (user bubbles)
        'obs-panel-alt': '#202123',      // assistant bubbles / header
        'obs-border': '#2f3136',         // subtle borders
        'obs-accent': '#f2c94c',         // yellow accent (buttons/badges)
      },
    },
  },
  plugins: [],
};
