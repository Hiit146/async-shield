"use client";
import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft, GitCommit, CheckCircle, XCircle, Clock, User as UserIcon, Coins } from "lucide-react";
import AuthWrapper, { User } from "@/components/AuthWrapper";

export default function RepoHistoryPage() {
  return (
    <AuthWrapper>
      {(user, logout, refreshUser) => <RepoHistoryContent user={user} />}
    </AuthWrapper>
  );
}

function RepoHistoryContent({ user }: { user: User }) {
  const params = useParams();
  const router = useRouter();
  const repoId = params.id as string;

  const [repo, setRepo] = useState<any>(null);
  const [commits, setCommits] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch repo details
        const reposRes = await fetch("http://localhost:8000/repos");
        const reposData = await reposRes.json();
        const currentRepo = reposData.find((r: any) => r.id === repoId);
        setRepo(currentRepo);

        // Fetch commits
        const commitsRes = await fetch(`http://localhost:8000/repos/${repoId}/commits`);
        const commitsData = await commitsRes.json();
        setCommits(commitsData);
      } catch (err) {
        console.error("Failed to fetch repo data", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [repoId]);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#020202] text-white flex items-center justify-center font-mono">
        <div className="animate-pulse">LOADING COMMIT HISTORY...</div>
      </div>
    );
  }

  if (!repo) {
    return (
      <div className="min-h-screen bg-[#020202] text-white flex flex-col items-center justify-center font-mono">
        <h1 className="text-2xl text-red-400 mb-4">REPOSITORY NOT FOUND</h1>
        <button onClick={() => router.back()} className="text-indigo-400 hover:underline flex items-center gap-2">
          <ArrowLeft size={16} /> GO BACK
        </button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#020202] text-white p-10 font-mono">
      <header className="mb-12 border-b border-white/5 pb-8">
        <button onClick={() => router.back()} className="text-gray-500 hover:text-white mb-6 flex items-center gap-2 transition-colors text-sm">
          <ArrowLeft size={16} /> Back to Dashboard
        </button>
        <div className="flex justify-between items-end">
          <div>
            <h1 className="text-3xl font-bold tracking-tighter text-indigo-400 flex items-center gap-3">
              {repo.name}
              <span className="text-xs bg-indigo-500/20 text-indigo-300 px-3 py-1 rounded-full border border-indigo-500/30">
                v{repo.version}
              </span>
            </h1>
            <p className="text-sm text-gray-500 mt-2 max-w-2xl">{repo.description}</p>
            <div className="flex items-center gap-4 mt-4 text-xs text-gray-400">
              <span className="flex items-center gap-1"><UserIcon size={14} /> Owner: {repo.owner}</span>
              <span>ID: {repo.id}</span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-4xl mx-auto">
        <h2 className="text-xl font-bold mb-6 flex items-center gap-2 border-b border-white/10 pb-4">
          <GitCommit className="text-gray-400" /> Commit History
        </h2>

        {commits.length === 0 ? (
          <div className="text-center py-12 bg-white/[0.02] border border-white/5 rounded-2xl">
            <GitCommit size={48} className="mx-auto text-gray-600 mb-4" />
            <p className="text-gray-400">No commits yet. Be the first to contribute!</p>
          </div>
        ) : (
          <div className="space-y-4">
            {commits.map((commit, idx) => (
              <div key={idx} className="bg-white/[0.02] border border-white/10 p-5 rounded-xl flex flex-col md:flex-row md:items-center justify-between gap-4 hover:bg-white/[0.04] transition-colors">
                <div className="flex items-start gap-4">
                  <div className="mt-1">
                    {commit.status.includes("Merged") ? (
                      <CheckCircle className="text-green-500" size={20} />
                    ) : (
                      <XCircle className="text-red-500" size={20} />
                    )}
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-bold text-white">{commit.client}</span>
                      <span className="text-xs text-gray-500">submitted an update</span>
                    </div>
                    <p className="text-sm text-gray-400">{commit.reason}</p>
                    <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                      <span className="flex items-center gap-1">
                        <Clock size={12} />
                        {new Date(commit.timestamp * 1000).toLocaleString()}
                      </span>
                      {commit.version_bump !== "None" && (
                        <span className="bg-blue-500/10 text-blue-400 px-2 py-0.5 rounded border border-blue-500/20">
                          {commit.version_bump}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center gap-4 md:flex-col md:items-end md:gap-2">
                  <div className={`text-sm font-bold px-3 py-1 rounded-full border ${commit.status.includes("Merged") ? 'bg-green-500/10 text-green-400 border-green-500/20' : 'bg-red-500/10 text-red-400 border-red-500/20'}`}>
                    {commit.status}
                  </div>
                  {commit.bounty > 0 && (
                    <div className="flex items-center gap-1 text-yellow-500 text-xs font-bold">
                      <Coins size={12} /> +{commit.bounty}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}