/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'soft-beige': '#f5f0e6', // warna beige lembut
        'deep-black': '#1a1a1a', // warna hitam pekat
      },
    },
  },
  plugins: [],
}

