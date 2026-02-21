# server/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import numpy as np
import json
import uuid
import os
import torch

from aggregator import RobustAggregator
from database import AsyncDatabase
from evaluator import Evaluator  
from models import RobustCNN, restore_1d_to_model

app = FastAPI()

# Enable CORS for Next.js Frontend (port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Components
db = AsyncDatabase("asyncshield.db")
aggregator = RobustAggregator()
evaluator = Evaluator() 

# Global Configuration
MAX_WEIGHTS = 500000  # Size for RobustCNN vector

# In-Memory Storage for weights (repo_id -> numpy array)
repo_weights_store = {}

# --- HELPER: INITIALIZE WEIGHTS ---
def get_initial_weights(model, target_size=MAX_WEIGHTS):
    weights = []
    state_dict = model.state_dict()
    for key in sorted(state_dict.keys()):
        weights.append(state_dict[key].cpu().numpy().flatten())
    flat_1d = np.concatenate(weights)
    if len(flat_1d) < target_size:
        padding = np.zeros(target_size - len(flat_1d))
        return np.concatenate([flat_1d, padding])
    return flat_1d[:target_size]

# --- REPOSITORY MANAGEMENT ---

@app.post("/create_repo")
async def create_repo(name: str = Form(...), description: str = Form(...), owner: str = Form(...)):
    """Orchestrator creates a new model project."""
    repo_id = str(uuid.uuid4())[:8]
    
    # Initialize with random RobustCNN brain
    initial_weights = get_initial_weights(RobustCNN())
    repo_weights_store[repo_id] = initial_weights
    
    db.create_repo(repo_id, name, description, owner)
    print(f"[ADMIN] New Repo Created: {name} ({repo_id})")
    return {"status": "success", "repo_id": repo_id}

@app.get("/repos")
def list_repos():
    """Returns all projects for the contributor dashboard."""
    return db.get_all_repos()

@app.get("/repos/{repo_id}/get_model")
def get_repo_model(repo_id: str):
    """Clients download current weights to start local training."""
    if repo_id not in repo_weights_store:
        # Recovery: If server restarted, re-init with random
        repo_weights_store[repo_id] = get_initial_weights(RobustCNN())
    
    repos = db.get_all_repos()
    repo_info = next((r for r in repos if r['id'] == repo_id), {"version": 1})
    
    return {
        "repo_id": repo_id,
        "version": repo_info["version"], 
        "weights": repo_weights_store[repo_id].tolist()
    }

# --- THE UNIFIED SUBMIT ENDPOINT (JSON File Upload) ---

@app.post("/repos/{repo_id}/submit_update")
async def submit_repo_update(
    repo_id: str, 
    client_id: str = Form(...), 
    client_version: int = Form(...), 
    file: UploadFile = File(...)
):
    print(f"\n[RECEIVED] Commit attempt for Repo: {repo_id} from {client_id}")

    # 1. Memory Safety
    if repo_id not in repo_weights_store:
        repo_weights_store[repo_id] = get_initial_weights(RobustCNN())

    # 2. Parse the JSON File
    try:
        contents = await file.read()
        data = json.loads(contents)
        delta = np.array(data["weights_delta"])
    except Exception as e:
        print(f"[ERROR] JSON Parse Failed: {e}")
        return {"status": "error", "message": "Invalid JSON format. Weights not found."}

    # 3. ZERO-TRUST EVALUATION
    global_weights = repo_weights_store[repo_id]
    real_delta_i, current_accuracy = evaluator.verify_update(global_weights, delta)
    
    print(f"[EVAL] ΔI: {real_delta_i*100:.4f}% | Target Acc: {current_accuracy*100:.2f}%")

    # FRAUD DETECTION: If accuracy drops by > 1.5%
    if real_delta_i < -0.015:
        db.add_commit(repo_id, client_id, "Rejected ❌", f"Fraud: Acc drop {abs(real_delta_i*100):.1f}%", "None", 0)
        return {"status": "rejected", "message": "Model poisoning detected by Zero-Trust."}

    # QUALITY GATE: If accuracy didn't improve at all
    if real_delta_i <= 0:
        db.add_commit(repo_id, client_id, "Rejected ❌", "No measurable improvement", "None", 0)
        return {"status": "rejected", "message": "Update did not improve model."}

    # 4. MATH: ADAPTIVE ASYNC AGGREGATION
    repos = db.get_all_repos()
    current_repo_v = next((r for r in repos if r['id'] == repo_id))['version']
    
    # Staleness + Intelligence Boost
    base_alpha = aggregator.calculate_staleness(current_repo_v, client_version)
    intel_boost = max(0, real_delta_i * 2.0)
    adaptive_trust = min(1.0, base_alpha + intel_boost)

    # 5. MERGE TO GLOBAL BRAIN
    repo_weights_store[repo_id] = global_weights + (aggregator.lr * adaptive_trust) * delta
    
    # Update Version and Bounty
    new_version = current_repo_v + 1
    db.update_repo_version(repo_id, new_version)
    bounty = 5 + int(real_delta_i * 10000)

    # Log GitHub-style Commit
    db.add_commit(
        repo_id, client_id, "Merged ✅", 
        f"Verified Improvement: {real_delta_i*100:.2f}%", 
        f"v{current_repo_v}->v{new_version}", bounty
    )

    print(f"[SUCCESS] Repo {repo_id} updated. Bounty: {bounty}")
    return {"status": "success", "bounty": bounty, "version": new_version}

# --- DASHBOARD & UTILS ---

@app.get("/dashboard_data")
def get_dashboard_data():
    """Comprehensive data for Admin dashboard."""
    repos = db.get_all_repos()
    # For demo, we just return commits for the first repo or all
    all_commits = []
    for r in repos:
        all_commits.extend(db.get_repo_commits(r['id']))
    
    return {
        "repos": repos,
        "commits": all_commits[:20] # Last 20 commits globally
    }

@app.get("/download_architecture")
def download_architecture():
    return FileResponse("models.py", media_type="text/x-python", filename="models.py")