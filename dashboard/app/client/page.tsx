// app/client/page.tsx
"use client";
import { useEffect, useState } from "react";
import { Search, UploadCloud, Coins } from "lucide-react";

export default function ClientDashboard() {
  const [repos, setRepos] = useState<any[]>([]);
  const [selectedRepo, setSelectedRepo] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);

  useEffect(() => {
    // Fetch available repos from server
    fetch("http://localhost:8000/repos")
      .then(res => res.json())
      .then(data => setRepos(data));
  }, []);

  const handleUpload = async () => {
    if (!selectedRepo || !file) return alert("Select a repo and a file!");
    
    const formData = new FormData();
    formData.append("client_id", "Contributor_99");
    formData.append("client_version", "1");
    formData.append("file", file);

    const res = await fetch(`http://localhost:8000/repos/${selectedRepo}/submit_update`, {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    alert(`Status: ${data.status} | Bounty: ${data.bounty || 0} 🪙`);
  };

  return (
    <div className="min-h-screen bg-[#050505] text-white p-10 font-sans">
      <header className="flex justify-between items-center mb-10 border-b border-white/10 pb-6">
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <Coins className="text-yellow-400"/> Contributor Hub
        </h1>
        <div className="px-4 py-2 bg-yellow-500/20 border border-yellow-500/30 rounded-full text-yellow-400 font-bold flex gap-2">
          Your Bounty Balance: <span>120 🪙</span>
        </div>
      </header>

      {/* SEARCH AND REPO LIST */}
      <div className="mb-8">
        <div className="relative max-w-md mb-6">
          <Search className="absolute left-3 top-2.5 text-gray-400" size={18}/>
          <input 
            type="text" 
            placeholder="Search open repositories..." 
            className="w-full bg-white/5 border border-white/20 rounded-full py-2 pl-10 pr-4 text-white focus:outline-none focus:border-indigo-500"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {repos.map(repo => (
            <div 
              key={repo.id} 
              onClick={() => setSelectedRepo(repo.id)}
              className={`p-5 rounded-xl border cursor-pointer transition-all ${selectedRepo === repo.id ? 'bg-indigo-900/40 border-indigo-400' : 'bg-white/5 border-white/10 hover:border-white/30'}`}
            >
              <h3 className="font-bold text-lg">{repo.name}</h3>
              <p className="text-sm text-gray-400 mt-1">{repo.description}</p>
              <div className="mt-4 flex justify-between items-center text-xs text-gray-500">
                <span>Owner: {repo.owner}</span>
                <span className="bg-green-900/50 text-green-400 px-2 py-1 rounded">v{repo.version}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* UPLOAD PANEL (Only shows if a repo is selected) */}
      {selectedRepo && (
        <div className="bg-indigo-950/20 p-8 rounded-xl border border-indigo-500/30 text-center max-w-2xl mx-auto mt-10">
          <UploadCloud className="mx-auto text-indigo-400 mb-4" size={48}/>
          <h2 className="text-2xl font-bold mb-2">Submit Model Update</h2>
          <p className="text-gray-400 mb-6">Upload your locally trained weights (.json format) for Repo ID: {selectedRepo}</p>
          
          <input 
            type="file" 
            accept=".json"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:bg-indigo-500 file:text-white hover:file:bg-indigo-600 cursor-pointer mx-auto max-w-xs mb-6"
          />
          
          <button 
            onClick={handleUpload}
            className="px-8 py-3 bg-indigo-600 hover:bg-indigo-500 rounded font-bold w-full max-w-xs transition-colors"
          >
            Submit Commit & Claim Bounty
          </button>
        </div>
      )}
    </div>
  );
}