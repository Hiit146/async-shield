import torch
import torch.optim as optim
from torchvision import datasets, transforms
from opacus import PrivacyEngine
import requests
import numpy as np
import uuid
import random
import time # Ensure time is imported for polling

from models import RobustCNN, restore_1d_to_model
from standardizer import WeightStandardizer

SERVER_URL = "http://localhost:8000"
CLIENT_ID = f"specialist-node-{uuid.uuid4().hex[:4]}"

def run_trainer():
    print(f"--- Starting Trainer 1: {CLIENT_ID} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1️⃣ Fetch Global State
    res = requests.get(f"{SERVER_URL}/get_model").json()
    global_weights_1d = np.array(res['weights'])
    current_version = res['version']

    # 2️⃣ Setup Model (RobustCNN)
    model = RobustCNN().to(device)
    model = restore_1d_to_model(model, global_weights_1d)

    # 3️⃣ Local Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    subset_indices = random.sample(range(len(dataset)), 5000)
    subset = torch.utils.data.Subset(dataset, subset_indices)
    train_loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)

    # 4️⃣ Privacy Engine
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=0.7, 
        max_grad_norm=1.0
    )

    # 5️⃣ Train
    epochs = 3
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

    # 6️⃣ Standardize and Push
    standardizer = WeightStandardizer(target_size=500000)
    updated_1d = standardizer.universal_standardize(model)
    delta = updated_1d - global_weights_1d

    payload = {
        "client_id": CLIENT_ID,
        "client_version": current_version,
        "weights_delta": delta.tolist()
    }

    print(f"[{CLIENT_ID}] Submitting to consensus pool...")
    response = requests.post(f"{SERVER_URL}/submit_update", json=payload).json()
    
    # 7️⃣ POLLING LOOP: Keep terminal active until TTL Watchdog or Threshold merges it
    if response.get('status') == 'queued':
        print(f"[{CLIENT_ID}] {response['message']}")
        print(f"[{CLIENT_ID}] Terminal active. Waiting for Consensus (Polling Dashboard every 3s)...")
        
        while True:
            time.sleep(3)
            try:
                stats = requests.get(f"{SERVER_URL}/dashboard_data").json()
                
                # Look for our client ID in the recent commits list
                my_commit = next((c for c in stats['commits'] if c['client_id'] == CLIENT_ID), None)
                
                if my_commit:
                    if "Merged" in my_commit['status']:
                        print(f"\n[{CLIENT_ID}] ✅ CONSENSUS REACHED! Status: {my_commit['status']}")
                        print(f"[{CLIENT_ID}] 💰 Final Bounty Earned: {my_commit['bounty']}")
                        break
                    elif "Rejected" in my_commit['status']:
                        print(f"\n[{CLIENT_ID}] ❌ CONSENSUS FAILED: Update was rejected.")
                        break
            except Exception as e:
                print(f"[{CLIENT_ID}] Error polling dashboard: {e}")
    else:
        # Immediate success or immediate rejection
        print(f"[{CLIENT_ID}] Server Response: {response}")

if __name__ == "__main__":
    run_trainer()