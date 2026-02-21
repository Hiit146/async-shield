// app/server/page.tsx
"use client";
import { useState } from "react";
import { Database, PlusCircle, Activity } from "lucide-react";

export default function ServerDashboard() {
  const [repoName, setRepoName] = useState("");
  const [desc, setDesc] = useState("");

  const handleCreateRepo = async (e: any) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("name", repoName);
    formData.append("description", desc);
    formData.append("owner", "Server_Admin_1");

    await fetch("http://localhost:8000/create_repo", {
      method: "POST",
      body: formData
    });
    alert("Repository Created Successfully!");
  };

  return (
    <div className="min-h-screen bg-[#050505] text-white p-10 font-sans">
      <header className="flex justify-between items-center mb-10 border-b border-white/10 pb-6">
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <Database className="text-rose-400"/> Model Owner Dashboard
        </h1>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
        {/* CREATE REPO FORM */}
        <div className="bg-white/5 p-6 rounded-xl border border-white/10">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <PlusCircle size={20}/> Create New Repository
          </h2>
          <form onSubmit={handleCreateRepo} className="space-y-4">
            <div>
              <label className="text-sm text-gray-400">Model Name</label>
              <input 
                type="text" 
                className="w-full bg-black border border-white/20 rounded p-2 mt-1 text-white" 
                placeholder="e.g. MNIST-V1-Global"
                onChange={(e) => setRepoName(e.target.value)}
              />
            </div>
            <div>
              <label className="text-sm text-gray-400">Description</label>
              <textarea 
                className="w-full bg-black border border-white/20 rounded p-2 mt-1 text-white" 
                placeholder="Detects handwritten digits..."
                onChange={(e) => setDesc(e.target.value)}
              />
            </div>
            {/* Mock Golden Dataset Upload for UI */}
            <div>
              <label className="text-sm text-gray-400">Attach Golden Dataset (.zip)</label>
              <input type="file" className="w-full text-sm text-gray-400 mt-1 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-white/10 file:text-white hover:file:bg-white/20"/>
            </div>
            <button type="submit" className="w-full bg-rose-600 hover:bg-rose-500 py-2 rounded font-bold transition-colors">
              Initialize Repository
            </button>
          </form>
        </div>

        {/* ANALYTICS SECTION */}
        <div className="bg-white/5 p-6 rounded-xl border border-white/10">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Activity size={20}/> Global Analytics
          </h2>
          <div className="space-y-4">
             <div className="p-4 bg-black rounded border border-white/10 flex justify-between">
                <span>Total Active Repos</span>
                <span className="text-rose-400 font-bold">1</span>
             </div>
             <div className="p-4 bg-black rounded border border-white/10 flex justify-between">
                <span>Total Client Commits</span>
                <span className="text-indigo-400 font-bold">24</span>
             </div>
             <div className="h-32 border border-dashed border-white/20 rounded flex items-center justify-center text-gray-500 text-sm">
               [ Commit History Graph Placeholder ]
             </div>
          </div>
        </div>
      </div>
    </div>
  );
}