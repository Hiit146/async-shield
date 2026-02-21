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

from aggregator import RobustAggregator
from database import AsyncDatabase
from evaluator import Evaluator  
from models import RobustCNN, restore_1d_to_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Components Initialization
db = AsyncDatabase("asyncshield.db")
aggregator = RobustAggregator()
evaluator = Evaluator() 

# 2. Global Configuration
MAX_WEIGHTS = 500000  # Size for RobustCNN

# IN-MEMORY STORE for Weights (repo_id -> numpy array)
repo_weights_store = {}

# --- HELPER FUNCTIONS ---

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

# --- REPO MANAGEMENT ENDPOINTS ---

@app.post("/create_repo")
async def create_repo(name: str = Form(...), description: str = Form(...), owner: str = Form(...)):
    """Server Owner creates a new Model Repository (GitHub 'New Repo' style)"""
    repo_id = str(uuid.uuid4())[:8]
    
    # Initialize weights with random RobustCNN weights (Heartbeat Init)
    initial_weights = get_initial_weights(RobustCNN())
    repo_weights_store[repo_id] = initial_weights
    
    # Save to SQLite
    db.create_repo(repo_id, name, description, owner)
    
    return {"status": "success", "repo_id": repo_id, "message": f"Repo '{name}' initialized."}

@app.get("/repos")
def list_repos():
    """Returns all available model repositories for clients to see"""
    return db.get_all_repos()

@app.get("/repos/{repo_id}/get_model")
def get_repo_model(repo_id: str):
    """Downloads weights for a specific repository"""
    if repo_id not in repo_weights_store:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    # Find current version from DB
    repos = db.get_all_repos()
    repo_info = next((r for r in repos if r['id'] == repo_id), {"version": 1})
    
    return {
        "repo_id": repo_id,
        "version": repo_info["version"], 
        "weights": repo_weights_store[repo_id].tolist()
    }

# --- UNIFIED UPDATE ENDPOINT (Zero-Trust + Adaptive Trust) ---

@app.post("/repos/{repo_id}/submit_update")
async def submit_repo_update(
    repo_id: str, 
    client_id: str = Form(...), 
    client_version: int = Form(...), 
    file: UploadFile = File(...)
):
    """
    Handles JSON file upload. 
    Applies Zero-Trust Evaluation, Adaptive Trust, and Staleness Penalty.
    """
    if repo_id not in repo_weights_store:
        return {"status": "error", "message": "Repo not found."}

    # 1. Parse File
    try:
        contents = await file.read()
        data = json.loads(contents)
        delta = np.array(data["weights_delta"])
    except Exception:
        return {"status": "error", "message": "Invalid JSON format."}

    global_weights = repo_weights_store[repo_id]
    
    # 2. ZERO-TRUST: EVALUATION
    # verify_update(old_weights, new_delta)
    real_delta_i, current_accuracy = evaluator.verify_update(global_weights, delta)
    
    # DEBUG LOG
    print(f"[REPO: {repo_id}] EVAL from {client_id}: ΔI: {real_delta_i*100:.4f}% | Acc: {current_accuracy*100:.2f}%")

    # FRAUD DETECTION (Threshold check)
    if real_delta_i < -0.015:
        db.add_commit(repo_id, client_id, "Rejected ❌", f"Fraud: Acc drop {abs(real_delta_i*100):.1f}%", "None", 0)
        return {"status": "rejected", "message": "Model poisoning detected by Zero-Trust."}

    # IF accuracy didn't improve, we log it but don't merge (Protects Global Brain)
    if real_delta_i <= 0:
        db.add_commit(repo_id, client_id, "Rejected ❌", "No improvement on Golden Set", "None", 0)
        return {"status": "rejected", "message": "Update did not improve model accuracy."}

    # 3. MATH: ADAPTIVE ASYNC TRUST
    # Use repo version for staleness math
    repos = db.get_all_repos()
    current_repo_v = next((r for r in repos if r['id'] == repo_id))['version']
    
    base_alpha = aggregator.calculate_staleness(current_repo_v, client_version)
    intel_boost = max(0, real_delta_i * 2.0)
    adaptive_trust = min(1.0, base_alpha + intel_boost)

    # 4. AGGREGATE & UPDATE
    # Apply change: W_new = W_old + (LR * Trust) * Delta
    repo_weights_store[repo_id] = global_weights + (aggregator.lr * adaptive_trust) * delta
    
    # Update Version in DB
    new_version = current_repo_v + 1
    db.update_repo_version(repo_id, new_version)
    
    # Calculate Bounty
    bounty = 5 + int(real_delta_i * 10000)

    # Log Commit in GitHub style
    db.add_commit(
        repo_id, 
        client_id, 
        "Merged ✅", 
        f"Verified Improvement: {real_delta_i*100:.2f}% | Trust: {adaptive_trust:.2f}", 
        f"v{current_repo_v}->v{new_version}", 
        bounty
    )

    return {
        "status": "success", 
        "bounty": bounty, 
        "new_version": new_version,
        "trust": f"{adaptive_trust:.2f}"
    }

# --- DASHBOARD ENDPOINTS ---

@app.get("/repos/{repo_id}/dashboard")
def get_repo_dashboard(repo_id: str):
    """Returns commit history and current state for a specific repo"""
    commits = db.get_repo_commits(repo_id)
    return {
        "repo_id": repo_id,
        "commits": commits
    }

@app.get("/download_architecture")
def download_architecture():
    return FileResponse("models.py", media_type="text/x-python", filename="models.py")