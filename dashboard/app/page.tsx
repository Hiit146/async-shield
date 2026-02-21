// app/page.tsx
"use client";
import { useRouter } from "next/navigation";
import { Web3HeroAnimated } from "@/components/ui/AnimatedWeb3Hero";

export default function LandingPage() {
  const router = useRouter();

  return (
    <div className="relative h-screen w-full">
      {/* Background Animated Template */}
      <div className="absolute inset-0 z-0">
        <Web3HeroAnimated />
      </div>

      {/* Floating Action Menu Over the Template */}
      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center pt-[20vh]">
        <h2 className="text-3xl font-bold text-white mb-8 tracking-wider bg-black/50 px-6 py-2 rounded-full backdrop-blur-md border border-white/10">
          Select Your Portal
        </h2>
        
        <div className="flex gap-6">
          <button 
            onClick={() => router.push('/server')}
            className="px-8 py-4 bg-rose-500/20 hover:bg-rose-500/40 border border-rose-500/50 rounded-xl backdrop-blur-md text-white font-semibold transition-all hover:scale-105"
          >
            👨‍💻 Login as Server (Model Owner)
          </button>
          
          <button 
            onClick={() => router.push('/client')}
            className="px-8 py-4 bg-indigo-500/20 hover:bg-indigo-500/40 border border-indigo-500/50 rounded-xl backdrop-blur-md text-white font-semibold transition-all hover:scale-105"
          >
            🛡️ Login as Client (Contributor)
          </button>
        </div>
      </div>
    </div>
  );
}