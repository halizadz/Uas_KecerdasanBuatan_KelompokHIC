/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "app.jsx",
    "./frontend/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          500: '#3b82f6', // blue-500
          600: '#2563eb'  // blue-600
        }
      },
    },
  },
  plugins: [
    require('daisyui'),
    require('@tailwindcss/forms')
  ],
  daisyui: {
    themes: ["light", "dark"],
  },
}