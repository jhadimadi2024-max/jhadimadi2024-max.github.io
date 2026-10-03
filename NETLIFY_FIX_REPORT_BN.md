# Netlify deploy fix report

এই সংস্করণে Netlify deploy আটকে দিতে পারে এমন প্রধান সমস্যাগুলো সংশোধন করা হয়েছে।

## করা পরিবর্তন

1. `npm run build` এখন শুধু Vite frontend build করে (`vite build`)। আগের build command একই সঙ্গে বড় Express `server.ts` bundle করছিল, যা Netlify static publish-এর জন্য প্রয়োজন নয়।
2. `netlify.toml`-এ build command, `dist` publish directory এবং Node 22 স্পষ্টভাবে সেট করা হয়েছে।
3. ZIP থেকে `.env` এবং runtime `data/admin_credentials.json` সরানো হয়েছে যাতে hosting secret scanner credential ধরে deploy বন্ধ না করে।
4. `.env.example` থেকে default admin credential সরানো হয়েছে।
5. source code-এর default admin password fallback সরানো হয়েছে; production-এ admin password environment variable দিয়ে দিতে হবে।
6. পুরোনো/stale `package-lock.json` এবং `bun.lock` সরানো হয়েছে। আগের `package-lock.json`-এ `bcryptjs`, `zod`-সহ package.json-এর কিছু dependency অনুপস্থিত ছিল। Netlify package.json থেকে clean dependency install করবে।
7. অব্যবহৃত `cloudinary` এবং অপ্রয়োজনীয় `@types/bcryptjs` dependency সরানো হয়েছে।
8. সব TypeScript source syntax-check করা হয়েছে এবং relative import path scan করা হয়েছে।

## Netlify Environment Variables

Project configuration → Environment variables-এ অন্তত দিন:

- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_ANON_KEY` অথবা `VITE_SUPABASE_PUBLISHABLE_KEY`

Server ব্যবহার করলে server-only secret (`ADMIN_PASSWORD`, `ADMIN_SECRET_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, `GEMINI_API_KEY` ইত্যাদি) browser/client-এ বা `VITE_` prefix দিয়ে দেবেন না।

## গুরুত্বপূর্ণ backend সীমাবদ্ধতা

এই project-এ `/api/...` endpoint-এর উপর অনেক feature নির্ভর করে এবং সেগুলো Express `server.ts`-এ আছে। Netlify-এর সাধারণ Vite static deploy শুধু frontend `dist` serve করবে; Express server নিজে থেকে চলবে না। Site deploy সফল হলেও admin authentication/AI/order/server API-এর কিছু feature কাজ করাতে backend আলাদা Node hosting-এ চালানো বা Netlify Functions-এ migrate করা প্রয়োজন।
