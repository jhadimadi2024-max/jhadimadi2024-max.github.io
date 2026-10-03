import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig, Plugin, loadEnv} from 'vite';

/**
 * Security Guard Plugin: Enforces that SUPABASE_SERVICE_ROLE_KEY,
 * ADMIN_SECRET_KEY, and other backend-only credentials are never exposed
 * to the frontend client bundle or prefixed with VITE_.
 */
function securitySecretGuardPlugin(): Plugin {
  return {
    name: 'security-secret-guard',
    configResolved() {
      const blockedPatterns = [
        /SERVICE_ROLE/i,
        /ADMIN_SECRET/i,
        /JWT_SECRET/i,
        /PRIVATE_KEY/i,
      ];

      for (const [key] of Object.entries(process.env)) {
        if (key.startsWith('VITE_')) {
          for (const pattern of blockedPatterns) {
            if (pattern.test(key)) {
              throw new Error(
                `[SECURITY ERROR] Privileged secret "${key}" detected with VITE_ prefix. ` +
                `Service role keys and admin secrets must NEVER be exposed to the client bundle. ` +
                `Keep them strictly on the server (e.g. server.ts).`
              );
            }
          }
        }
      }
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const fallbackUrl = 'https://dwhsqftllkximhfvwqak.supabase.co';
  const fallbackAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR3aHNxZnRsbGt4aW1oZnZ3cWFrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODk3MzAyNzEsImV4cCI6MjEwNTMwNjI3MX0.GbceleQmKhRfSzE-c_Bq3fh-YA7I4oZI1fGCsU-SaPI';

  const supabaseUrl = env.VITE_SUPABASE_URL || env.NEXT_PUBLIC_SUPABASE_URL || process.env.VITE_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL || fallbackUrl;
  const supabaseAnonKey = env.VITE_SUPABASE_PUBLISHABLE_KEY || env.VITE_SUPABASE_ANON_KEY || env.NEXT_PUBLIC_SUPABASE_ANON_KEY || process.env.VITE_SUPABASE_PUBLISHABLE_KEY || process.env.VITE_SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || fallbackAnonKey;

  return {
    plugins: [react(), tailwindcss(), securitySecretGuardPlugin()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    define: {
      'process.env.VITE_SUPABASE_URL': JSON.stringify(supabaseUrl),
      'process.env.VITE_SUPABASE_ANON_KEY': JSON.stringify(supabaseAnonKey),
      'process.env.NEXT_PUBLIC_SUPABASE_URL': JSON.stringify(supabaseUrl),
      'process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY': JSON.stringify(supabaseAnonKey),
      'import.meta.env.VITE_SUPABASE_URL': JSON.stringify(supabaseUrl),
      'import.meta.env.VITE_SUPABASE_ANON_KEY': JSON.stringify(supabaseAnonKey),
      'import.meta.env.NEXT_PUBLIC_SUPABASE_URL': JSON.stringify(supabaseUrl),
      'import.meta.env.NEXT_PUBLIC_SUPABASE_ANON_KEY': JSON.stringify(supabaseAnonKey),
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modifyâfile watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
    },
  };
});
