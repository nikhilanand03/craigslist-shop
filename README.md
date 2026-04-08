# Craigslist Shop: A Negotiation Benchmark for LLM Agents

A benchmark environment for training and evaluating LLM agents on price negotiation. The agent plays the role of a **seller** on Craigslist, tasked with selling items to diverse, realistic buyer personas extracted from real human-to-human negotiation data.

## What is this?

Craigslist Shop is built on the [stanfordnlp/craigslist_bargains](https://huggingface.co/datasets/stanfordnlp/craigslist_bargains) dataset — 6,682 real buyer-seller conversations collected via Mechanical Turk. From these conversations, we extracted 2,000 unique buyer personas (1,200 train / 800 test), each with:

- A **task** (what the buyer wants and why)
- A **personality** (negotiation style, temperament, tactics)
- A **response style** (tone, formality, message length, verbal habits)
- The **source conversation** from the original dataset as behavioral grounding

Each episode is a single negotiation. The agent receives the item details (title, description, category, listed price) and must negotiate with a buyer LLM that role-plays one of these personas. The buyer decides autonomously when to accept a price or walk away — there are no fixed thresholds.

**Reward** = `sale_price / listed_price`
- 1.0 = sold at full listed price
- 0.5 = sold at half price
- 0.0 = buyer walked away (no sale)

The agent's goal: **maximize price retention across diverse buyer types.**

## Setup

### Prerequisites

- Python 3.10+
- Docker Desktop
- Azure OpenAI API keys (for the customer LLM)

### Installation

```bash
# Clone the repo
git clone https://github.com/nikhilanand03/burger-shack.git
cd burger-shack

# Create conda environment
conda create -n craigslist_shop python=3.11
conda activate craigslist_shop

# Install dependencies
pip install -r requirements.txt
```

### Configure API Keys

Create a `key.json` in the repo root. You can use either **OpenAI** or **Azure OpenAI**:

**Option A: OpenAI**
```json
{
    "openai_api_key": "sk-...",
    "openai_model": "gpt-4o"
}
```

**Option B: Azure OpenAI**
```json
{
    "azure_openai_endpoint": "https://your-endpoint.openai.azure.com/",
    "azure_openai_api_key": "your-api-key"
}
```

If both are present, OpenAI takes priority over Azure.

You can also set environment variables instead of (or in addition to) `key.json`:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o"          # optional, defaults to gpt-4o

# Or Azure OpenAI
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_MODEL="gpt-4o"    # optional, defaults to gpt-4o
```

Values in `key.json` take priority over environment variables. If neither is set, the scripts will show an error.

Copy `key.json` into the environment folder for Docker access:
```bash
cp key.json craigslist_shop/key.json
```

> `key.json` is already in `.gitignore` and will not be committed.

### Build and Run the Docker Container

```bash
cd craigslist_shop
docker build -t craigslist_shop_env:latest -f server/Dockerfile .
docker run -p 8000:8000 craigslist_shop_env:latest
```

The server will be available at `http://localhost:8000`. You can verify it at `http://localhost:8000/docs` to see the FastAPI endpoints.

To stop the container:
```bash
docker stop $(docker ps -q --filter ancestor=craigslist_shop_env:latest)
```

### Running Without Docker

You can also run the server directly:
```bash
cd craigslist_shop
python -m server.app
```

## Demo

The demo runs built-in seller strategies against the environment and produces evaluation metrics and plots. This is the quickest way to see the environment in action.

### Run a Single Strategy

```bash
cd craigslist_shop

# List available strategies
python test_agent.py --list-strategies

# Run 10 episodes sequentially from the test set
python test_agent.py --strategy skilled_seller --episodes 10

# Run 10 randomly sampled episodes from the test set
python test_agent.py --strategy skilled_seller --episodes 10 --sample

# Save results to a custom directory
python test_agent.py --strategy skilled_seller --episodes 10 --suffix experiment_v1

# Suppress per-turn output, show only summary
python test_agent.py --strategy pushover --episodes 20 --quiet
```

### Sample Strategies

| Strategy | Description |
|----------|-------------|
| `pushover` | Always accepts the buyer's first offer |
| `full_price` | Never discounts, rejects if buyer pushes back |
| `skilled_seller` | Discounts up to 10%, uses persuasion to close |
| `haggler` | Enjoys negotiation, willing to go down to 70% |
| `random` | Chaotic, unpredictable pricing each turn |

### Run the Full Demo (Sample Strategies)

Use `run_analysis.sh` to evaluate all sample strategies and generate comparison plots in one go:

```bash
# Run all 5 strategies x 10 episodes, generate plots
bash run_analysis.sh

# Or with a custom suffix for the run
bash run_analysis.sh experiment_v2
```

This will:
1. Run each sample strategy (`pushover`, `full_price`, `skilled_seller`, `haggler`, `random`) for 10 episodes on the test set
2. Save per-strategy results to `runs_{suffix}/`
3. Generate `analysis/strategy_evaluation_{suffix}.png` with three charts: average reward (with std error bars), sale rate, and price retention per strategy
4. Print a summary table to the terminal

### Demo Output

Results are saved to `runs/` (or `runs_{suffix}/`) as JSON files containing:
- Per-episode metrics (reward, outcome, sale price, listed price, turns)
- Full conversation history for each episode
- Aggregate statistics (avg reward, sale rate, price retention, per-category breakdown)

## Building Your Own Agent

Your agent interacts with the environment via HTTP. Each step, the agent receives an observation and must return an action:

**Observation fields:**
- `item_title`, `item_description`, `item_category`, `listed_price` — what you're selling
- `customer_message` — what the buyer just said
- `current_offer_price` — your last offered price
- `turn` — current negotiation turn number
- `conversation_history` — full dialogue so far

**Action fields:**
- `message` (str) — your natural language response to the buyer
- `price` (float, optional) — the price you're offering

The agent can only negotiate — it sends messages and prices. The buyer decides the outcome. The episode ends when:
- The buyer accepts a price (`[ACCEPT $X.XX]`)
- The buyer walks away (`[WALKAWAY]`)
- The hard turn cap (20) is reached

## Generating Personas

The persona extraction pipeline uses GPT-4o to analyze real Craigslist conversations and extract buyer personas. To regenerate or modify:

```bash
# Full extraction (calls GPT, takes ~2 hours with 5 workers)
python extract_personas.py

# Backfill source conversations into existing data (no GPT calls)
python extract_personas.py --fill-conversations

# Rebuild system prompts from persona fields (no GPT calls)
python extract_personas.py --rebuild-prompts
```

## Real-World Applications

This benchmark captures a fundamental problem in commercial AI: **how do you maximize transaction value while keeping the counterparty engaged?** This applies to:

- **Subscription sales** — An agent selling SaaS subscriptions has no inventory constraint, but must negotiate pricing with prospects. Any subscription is better than no subscription, but every dollar of discount is lost recurring revenue. The agent must read buyer intent and hold price where possible.

- **Real estate and auto sales** — High-value negotiations where the agent must justify price through features, condition, and market positioning while handling objections and emotional tactics.

- **Marketplace pricing** — Platforms like eBay, Facebook Marketplace, and Craigslist where sellers interact with diverse buyers, each with different budgets and negotiation styles.

- **Customer retention** — When a customer threatens to cancel, the agent must negotiate retention offers — giving away as little as possible while preventing churn.

## Next Steps

The current setup is one-item-one-customer: each episode is an isolated negotiation. Natural extensions include:

- **Inventory management** — The agent has N copies of an item and faces a queue of customers. Selling cheap early depletes stock that could go to better-paying buyers later. This introduces opportunity cost — the agent must balance closing deals now vs. holding out for higher offers.

- **Product catalog** — The agent manages multiple items with different margins. It must decide which items to discount, which to hold firm on, and how to cross-sell.

- **Multi-round markets** — Customers who walk away might come back. The agent must model repeat interactions and long-term value.

These extensions build directly on the single-negotiation skills this benchmark develops. An agent that can maximize price retention across diverse buyer personas is the foundation for all of the above.

## Project Structure

```
burger-shack/
├── README.md                  # This file
├── key.json                   # Azure OpenAI keys (gitignored)
├── extract_personas.py        # Persona extraction pipeline
├── craigslist_shop/           # OpenEnv environment
│   ├── server/
│   │   ├── app.py             # FastAPI server
│   │   ├── craigslist_shop_environment.py  # Core environment logic
│   │   ├── scoring.py         # Reward computation
│   │   └── state_machine.py   # Episode phase tracking
│   ├── tasks/
│   │   ├── train.json         # 1200 training personas
│   │   └── test.json          # 800 test personas
│   ├── models.py              # Action/Observation schemas
│   ├── client.py              # Environment client
│   ├── test_agent.py          # Evaluation script with built-in strategies
│   └── run.py                 # CLI runner for custom agents
└── runs/                      # Evaluation results (gitignored)
```
