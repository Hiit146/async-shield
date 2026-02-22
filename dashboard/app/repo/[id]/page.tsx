"use client";
import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft, GitCommit, CheckCircle, XCircle, Clock, User as UserIcon, Coins, Activity } from "lucide-react";
import AuthWrapper, { User } from "@/components/AuthWrapper";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

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
  const [chartData, setChartData] = useState<any[]>([]);
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
        const commitsArray = Array.isArray(commitsData) ? commitsData : [];
        setCommits(commitsArray);

        // Process data for chart (reverse to chronological order)
        const mergedCommits = commitsArray
          .filter(c => c.status.includes("Merged"))
          .reverse();
        
        const data = mergedCommits.map((c, index) => ({
          version: `v${index + 2}`, // Assuming initial is v1
          accuracy: c.accuracy ? parseFloat((c.accuracy * 100).toFixed(2)) : 0,
          client: c.client
        }));
        
        // Add initial v1 point if we have data
        if (data.length > 0 && data[0].accuracy > 0) {
          // Estimate v1 accuracy based on the first improvement
          const firstImpMatch = mergedCommits[0].reason.match(/Imp: ([\d.]+)%/);
          const firstImp = firstImpMatch ? parseFloat(firstImpMatch[1]) : 0;
          data.unshift({
            version: 'v1',
            accuracy: parseFloat((data[0].accuracy - firstImp).toFixed(2)),
            client: 'Initial'
          });
        }
        
        setChartData(data);
      } catch (err) {
        console.error("Failed to fetch repo data", err);
        setCommits([]);
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
        {/* ACCURACY CHART */}
        {chartData.length > 0 && (
          <div className="mb-12 bg-white/[0.02] border border-white/10 p-6 rounded-2xl">
            <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
              <Activity className="text-indigo-400" /> Global Model Accuracy
            </h2>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
                  <XAxis dataKey="version" stroke="#ffffff50" tick={{ fill: '#ffffff50', fontSize: 12 }} />
                  <YAxis stroke="#ffffff50" tick={{ fill: '#ffffff50', fontSize: 12 }} domain={['auto', 'auto']} tickFormatter={(val) => `${val}%`} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#020202', borderColor: '#ffffff20', borderRadius: '8px' }}
                    itemStyle={{ color: '#818cf8' }}
                    formatter={(value: any) => [`${value}%`, 'Accuracy']}
                    labelStyle={{ color: '#ffffff80', marginBottom: '4px' }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="accuracy" 
                    stroke="#818cf8" 
                    strokeWidth={3}
                    dot={{ r: 4, fill: '#020202', stroke: '#818cf8', strokeWidth: 2 }}
                    activeDot={{ r: 6, fill: '#818cf8', stroke: '#020202', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        <h2 className="text-xl font-bold mb-6 flex items-center gap-2 border-b border-white/10 pb-4">
          <GitCommit className="text-gray-400" /> Commit History
        </h2>

        {commits.length === 0 ? (
          <div className="text-center py-12 bg-white/[0.02] border border-white/5 rounded-2xl">
            <GitCommit size={48} className="mx-auto text-gray-600 mb-4" />
            <p className="text-gray-400">No commits yet. Be the first to contribute!</p>
          </div>
        ) : (
          <div className="space-y-4 relative before:absolute before:inset-0 before:ml-5 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-white/10 before:to-transparent">
            {commits.map((commit, idx) => (
              <div key={idx} className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
                {/* Timeline dot */}
                <div className="flex items-center justify-center w-10 h-10 rounded-full border-4 border-[#020202] bg-white/[0.02] group-hover:bg-white/[0.05] text-white shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 z-10 transition-colors">
                  {commit.status.includes("Merged") ? (
                    <CheckCircle className="text-green-500" size={16} />
                  ) : (
                    <XCircle className="text-red-500" size={16} />
                  )}
                </div>
                
                {/* Card */}
                <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] bg-white/[0.02] border border-white/10 p-5 rounded-xl hover:bg-white/[0.04] transition-colors">
                  <div className="flex flex-col gap-3">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-bold text-white">{commit.client}</span>
                          <span className="text-xs text-gray-500">submitted an update</span>
                        </div>
                        <p className="text-sm text-gray-400">{commit.reason}</p>
                      </div>
                      <div className={`text-xs font-bold px-2 py-1 rounded-full border whitespace-nowrap ${commit.status.includes("Merged") ? 'bg-green-500/10 text-green-400 border-green-500/20' : 'bg-red-500/10 text-red-400 border-red-500/20'}`}>
                        {commit.status}
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between mt-2 pt-3 border-t border-white/5">
                      <div className="flex items-center gap-3 text-xs text-gray-500">
                        <span className="flex items-center gap-1">
                          <Clock size={12} />
                          {new Date(commit.timestamp * 1000).toLocaleString()}
                        </span>
                        {commit.version_bump !== "None" && (
                          <span className="bg-blue-500/10 text-blue-400 px-2 py-0.5 rounded border border-blue-500/20 font-mono">
                            {commit.version_bump}
                          </span>
                        )}
                      </div>
                      {commit.bounty > 0 && (
                        <div className="flex items-center gap-1 text-yellow-500 text-xs font-bold">
                          <Coins size={12} /> +{commit.bounty}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}