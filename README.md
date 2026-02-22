# 🛡️ AsyncShield: Privacy-Preserving Asynchronous Federated Learning

AsyncShield is a robust, privacy-preserving federated learning system designed to enable multiple clients to collaboratively train machine learning models without ever sharing their raw, sensitive data. 

Built for real-world, decentralized environments, AsyncShield natively supports asynchronous updates, handles unreliable or malicious clients (Byzantine fault tolerance), and ensures high convergence rates even on highly heterogeneous (Non-IID) datasets.

---

## ✨ Mandatory Features

The system is architected to fulfill the following core capabilities:

- **Support for asynchronous client updates**: Clients can pull the global model, train locally, and push updates at their own pace without waiting for synchronous rounds.
- **Robust aggregation algorithms**: Implements dynamic aggregation strategies including Trimmed Mean, Cosine Similarity, and Golden Dataset evaluation to mitigate poisoning attacks.
- **Mechanism to detect/mitigate malicious or noisy clients**: Automatically identifies and isolates bad actors submitting degraded or poisoned weights.
- **Clear demonstration of privacy preservation**: Ensures raw data never leaves the client device, utilizing local differential privacy techniques before transmission.
- **Performance evaluation**: Tracks and visualizes training stability, convergence improvements, and global model accuracy over time via a dedicated dashboard.
- **Working prototype with reproducible results**: Includes a complete pipeline with a Python backend server, Python training clients, and a Next.js monitoring dashboard.

---

## 🔄 The Entire Process: How AsyncShield Works

1. **Initialization**: The central server initializes a global model and a trusted "Golden Dataset" for baseline evaluation.
2. **Client Pull**: Clients asynchronously request the latest global model weights from the server.
3. **Local Training**: Clients train the model on their local, private data.
4. **Privacy Shielding**: Clients apply privacy-preserving transformations (e.g., gradient clipping, noise addition) to their model updates.
5. **Asynchronous Push**: Clients push their updated weights to the server's aggregation queue (buffer).
6. **Dynamic Aggregation**: The server processes the queue based on its current size (see *Dynamic Aggregation Strategy* below).
7. **Global Update & Scoring**: If the aggregated update is deemed beneficial, the global model is updated, the version is bumped, and the contributing clients are awarded a bounty/score.

---

## 🧠 Core Mechanisms & Formulas

### 1. Dynamic Aggregation Strategy (The Buffer System)

To handle asynchronous updates efficiently while maintaining security against malicious clients, AsyncShield uses a dynamic queue/buffer system. The aggregation method adapts based on the number of pending updates in the buffer:

#### **Case A: Buffer Size == 1 (Golden Dataset Evaluation)**
When only a single client update is in the queue, the server cannot compare it against peers. Instead, it evaluates the update directly against a small, trusted **Golden Dataset** held on the server.
*   **Decision Rule**: If the new model's accuracy on the golden dataset ($Acc_{new}$) is greater than or equal to the current global model's accuracy ($Acc_{global}$) minus a small tolerance margin ($\epsilon$), the update is accepted.
*   **Formula**: 
    $$ Accept \iff Acc_{new} \ge Acc_{global} - \epsilon $$

#### **Case B: Buffer Size == 2 (Cosine Similarity)**
When two updates are in the buffer, the server calculates the Cosine Similarity between the two weight update vectors ($\Delta W_1$ and $\Delta W_2$). 
*   **Decision Rule**: If the updates point in the same general direction (similarity > threshold $\tau$), they are averaged and applied. If they diverge significantly, they are flagged as potentially noisy/malicious and fall back to individual Golden Dataset evaluation.
*   **Formula**:
    $$ S_C(\Delta W_1, \Delta W_2) = \frac{\Delta W_1 \cdot \Delta W_2}{\|\Delta W_1\| \|\Delta W_2\|} $$
    $$ W_{agg} = W_{global} + \frac{\Delta W_1 + \Delta W_2}{2} \quad \text{if } S_C > \tau $$

#### **Case C: Buffer Size $\ge$ 3 (Trimmed Mean Aggregation)**
When three or more updates are available, the system uses a robust **Trimmed Mean** algorithm. This effectively mitigates Byzantine attacks by discarding extreme values (potential poisoned updates) before averaging.
*   **Process**: For each parameter in the model, sort the values submitted by all $N$ clients. Remove the top $k$ and bottom $k$ values, and compute the mean of the remaining values.
*   **Formula**:
    $$ w_{agg} = \frac{1}{N - 2k} \sum_{i=k+1}^{N-k} w_{(i)} $$
    *(where $w_{(i)}$ represents the sorted weights for a specific parameter)*

### 2. Scoring and Bounty Calculation

Clients are incentivized to provide high-quality updates. The score (or bounty) awarded to a client is proportional to the improvement their update brings to the global model.

*   **Accuracy Delta**: 
    $$ \Delta Acc = Acc_{new} - Acc_{global} $$
*   **Bounty Formula**: 
    $$ Bounty = \max(0, \alpha \times \Delta Acc + \beta) $$
    *(where $\alpha$ is a scaling multiplier and $\beta$ is a base reward for successful participation)*

### 3. Privacy Preservation (Differential Privacy)

To ensure that the server cannot reverse-engineer a client's raw data from their weight updates, clients apply Local Differential Privacy (LDP) before uploading.

*   **Gradient Clipping**: Limits the maximum influence of any single data point.
*   **Noise Addition**: Adds Gaussian or Laplacian noise to the clipped weights.
*   **Formula**:
    $$ W_{upload} = W_{trained} + \mathcal{N}(0, \sigma^2 I) $$

---

## 📂 Repository Structure

*   `/server`: The central aggregator. Contains the backend API, database, evaluation logic, and the dynamic buffer implementation.
*   `/client`: The federated learning nodes. Includes standard trainers, privacy wrappers, and simulated malicious clients (`bad_client.py`) for testing robustness.
*   `/dashboard`: A Next.js web application for real-time monitoring of global model accuracy, commit history, client leaderboards, and bounty distributions.
*   `/data` & `/test_data`: Sample datasets used for local training and the server's golden dataset.

---

## 🚀 Getting Started

### 1. Start the Server
```bash
cd server
pip install -r ../requirements.txt
uvicorn main:app --reload
```

### 2. Start the Dashboard
```bash
cd dashboard
npm install
npm run dev
```

### 3. Run Clients
```bash
cd client
python trainer.py
```