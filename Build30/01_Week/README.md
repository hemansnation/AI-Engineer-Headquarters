# Build30 Season 1: Day-Wise Topics

**AI Engineer Roadmap · Week 0 plus 30 days · Himanshu Ramchandani · @hemansnation**

One build a day. Eight live sessions. Five ship days. Three read days.

This file lists every topic for every day, in order, taken from the Build30 roadmap PDF. Use it as the single index for the season.

---

## How to Read This File

Every day has the same shape: a title, a type, one line saying what you build, the topics, and the flow of the build in one line.

| Type | What it means |
|---|---|
| BUILD | An ordinary build day. The roadmap leaves these untagged. |
| LIVE | 90 minute live build session, Wed and Fri, 7:30 PM IST |
| SHIP | Public ship day. Your work goes out. |
| READ | Research deep dive. A paper plus a reflection. |

Every day, without exception: **1 build, 3 interview questions, 1 technical problem, 1 system design pattern, 1 AI-DSA topic.**

Over 30 days that adds up to 30 builds, 90 interview questions, 30 technical problems, 30 system design patterns, and 30 AI-DSA topics.

---

## The Four Weeks

| Week | Theme | What it covers | Days |
|---|---|---|---|
| 0 | Zero-to-One | terminal, Python, APIs, vector maths. Free, Sep 25 to 27 | 4 modules |
| 1 | Foundations | tokenization, embeddings, your first vector store | Days 01 to 07 |
| 2 | Retrieval | chunking, RAG 2.0, hybrid search, tools | Days 08 to 14 |
| 3 | Agents | ReAct loops, sandboxes, guardrails, evals | Days 15 to 21 |
| 4 | Production | serving, FDE skills, capstone system design | Days 22 to 30 |

---

## All 30 Days at a Glance

| Day | Topic | Type |
|---|---|---|
| 0.1 | Terminal & Git Without Fear | Week 0 |
| 0.2 | Python for AI Engineering | Week 0 |
| 0.3 | How Software Talks to Software | Week 0 |
| 0.4 | Vector Mathematics Without Calculus | Week 0 |
| 01 | Dev Environment & First LLM Call | BUILD |
| 02 | Temperature, Logits & Formatting | BUILD |
| 03 | Tokenization & Character Blindness | LIVE |
| 04 | Embeddings: Text into Numerical Meaning | BUILD |
| 05 | Building an In-Memory Vector Store | LIVE |
| 06 | Ship Day: Week 1 Public Proof | SHIP |
| 07 | Read Day: Systems Reflection | READ |
| 08 | Document Chunking & Token Budget | BUILD |
| 09 | Your First Complete RAG Pipeline | BUILD |
| 10 | Hybrid Search & Reciprocal Rank Fusion | LIVE |
| 11 | Structured Outputs & JSON Extraction | BUILD |
| 12 | Tool Calling & Function Execution | LIVE |
| 13 | Ship Day: Production RAG 2.0 Showcase | SHIP |
| 14 | Read Day: RAG Failure Modes | READ |
| 15 | The ReAct Agent Loop From Scratch | BUILD |
| 16 | Isolated Code Execution Sandboxing | BUILD |
| 17 | Multi-Agent: Planner & Reviewer | LIVE |
| 18 | Automated Evals & Golden Sets | BUILD |
| 19 | Guardrails & Prompt Injection Defense | LIVE |
| 20 | Ship Day: Agentic System Showcase | SHIP |
| 21 | Read Day: The ReAct Paradigm | READ |
| 22 | High-Throughput APIs with FastAPI | BUILD |
| 23 | Hardware Sizing & Memory Calculations | BUILD |
| 24 | Real-Time Streaming with SSE | LIVE |
| 25 | The FDE Client Discovery & Scoping | BUILD |
| 26 | Safe Database Integration: Text-to-SQL | BUILD |
| 27 | Ship Day: Containerize & Deploy | SHIP |
| 28 | Capstone: The 10,000-User Blueprint | LIVE |
| 29 | Portfolio Assembly & Case Study | BUILD |
| 30 | Grand Showcase & Graduation | SHIP |

---

## Week 0: Zero-to-One

**Sep 25 to 27 · free**

Start from wherever you actually are. Zero prior AI experience required.

Day 0 Orientation is live with Himanshu on Friday, September 25 at 7:30 PM IST.

### Module 0.1: Terminal & Git Without Fear

15 core commands plus fixing real errors.

- Navigating a filesystem from the command line
- Creating, moving, copying, and deleting files without a GUI
- Reading an error message instead of pasting it into Google
- git init, add, commit, and what each one actually moves
- Branches, and why you will use them every day for 30 days
- Pushing to a remote, and resolving your first rejected push
- What to do when Git says something you do not understand

**The flow:** working dir (your editor) → staging (git add) → local repo (git commit) → remote (git push) → git pull, and you are back at the start

### Module 0.2: Python for AI Engineering

Lists, dictionaries, functions, and JSON.

- Lists, indexing, slicing, and list comprehensions
- Dictionaries, key lookup, and nested structures
- Writing functions with parameters, defaults, and return values
- Reading and writing JSON, and why every API speaks it
- Virtual environments and pip install
- f-strings and formatting output you can actually read
- The errors you will hit most: KeyError, TypeError, IndentationError

**The flow:** API response (raw text) → json.loads() → dict → data["choices"] → list → [0]

### Module 0.3: How Software Talks to Software

APIs, HTTP requests, and .env security.

- What an API is, in terms of request and response
- HTTP verbs, status codes, and headers
- Making your first request with requests
- Reading an API response and pulling one field out of it
- API keys, and why they never go in your code
- .env files, python-dotenv, and .gitignore
- Rate limits, timeouts, and retry behaviour

**The flow:** your script (requests.post()) → headers (Authorization) → server (the API) → response (status + body)

### Module 0.4: Vector Mathematics Without Calculus

Intuitive spatial geometry.

- A vector is a list of numbers, and nothing more
- Plotting meaning in 2D before scaling to 384 dimensions
- Direction versus magnitude, and why direction carries the meaning
- Dot product as how aligned are these two
- Magnitude as the length of the arrow
- Cosine similarity as one divide, and reading the score
- Why no calculus is required for anything in this cohort

**The flow:** dot product (multiply, add) + magnitude (square, add, root) → one divide, dot / (|a| x |b|) → a score from -1.0 to 1.0

### What Week 0 closes

- No terminal becomes 15 commands, and the fear gone
- No Python becomes lists, dicts, functions, JSON
- No API experience becomes your first request, keys kept safe
- No maths becomes vectors, and no calculus anywhere

Then you are ready for Day 1.

---

## Week 1: Foundations

**Days 01 to 07 · tokenization, embeddings, your first vector store**

### Day 01: Development Environment, First LLM Call & Token Tracker

`BUILD` · Week 1 · Foundations

A working environment, your first completion call, and a log of every token you spend.

- Project structure, virtual environment, and dependency file
- Getting and storing an API key safely
- Your first completion call, request and response
- Reading the usage object: prompt tokens, completion tokens, total
- Writing a token tracker that logs every call to a file
- Estimating cost per call from token counts
- Handling the first failure: bad key, rate limit, timeout

**The flow:** your script (main.py) → request (prompt + key) → the model (completion) → response (text + usage) → token log (append to file)

**Season 1 note:** this season's Day 1 build uses the free Gemini API, so the usage object has Gemini's field names: `total_input_tokens`, `total_output_tokens`, `total_thought_tokens`, and `total_tokens`. Thinking tokens are billed at the output price, so the tracker logs them separately. Day 1 was also taught live this season, which is why the season has nine live sessions and the roadmap tags eight.

### Day 02: Temperature, Logits & Response Formatting

`BUILD` · Week 1 · Foundations

Same prompt, different settings, and a harness that shows you exactly what changed.

- What a logit is, before softmax turns it into a probability
- Temperature, and what it actually does to the distribution
- Top-p and top-k sampling, and when each matters
- Determinism, seeds, and why the same prompt gives different answers
- Controlling output shape through the prompt
- Stop sequences and max tokens
- Building a side-by-side comparison output you can read at a glance

**The flow:** prompt (your text) → logits (raw scores) → temperature (reshapes them) → sampled token (the output). The same prompt at 0.0 (deterministic), 0.7 (the default), and 1.2 (creative, wilder)

### Day 03: Tokenization & Character Blindness

`LIVE` · Week 1 · Foundations

Build a tokenizer with encode and decode, then prove why the model cannot count.

- Characters, words, and sub-words as three tokenization strategies
- Building a vocabulary from a corpus
- encode and decode as inverse operations
- Byte-Pair Encoding, and why it beats word-level splitting
- Why an LLM fails to count the r's in strawberry
- Why Hindi and Japanese consume more tokens than English
- Token cost as a direct function of your tokenizer choice

**The flow:** text ("strawberry") → tokenizer (split to pieces) → token ids → decode (back to text). The model sees str | aw | berry, three tokens, never three letters

### Day 04: Embeddings: Turning Text into Numerical Meaning

`BUILD` · Week 1 · Foundations

An embedding pipeline that proves similar meaning produces similar numbers.

- Why a machine cannot compare English directly
- Embeddings as ASCII with the relationship preserved
- Calling an embedding model and reading the output shape
- Fixed dimensionality, and why the length never changes
- Comparing two embeddings and reading the score
- Choosing an embedding model: dimensions, cost, and speed
- Batch embedding and caching, so you stop paying twice

**The flow:** sentence (raw text) → embedding model (one call) → vector (384 floats) → cache (never pay twice)

### Day 05: Building an In-Memory Vector Store

`LIVE` · Week 1 · Foundations

A file-backed TinyVectorDB with add_document, query, and save_to_disk.

- Storing text, vector, and metadata together as one record
- add_document and what a document ID buys you
- Brute-force nearest neighbour search across N vectors
- Top-K selection by sorting, and by priority queue
- Why metadata alongside the vector is not optional
- Persisting to disk and loading back without re-embedding
- Where an in-memory store stops being viable

**The flow:** add() (text + vector + meta) → the store (a Python list) → query (score every item) → top_k (sort, then slice). save_to_disk means the next run skips the embedding entirely

### Day 06: Public Ship Day: Week 1 Review & Public Proof

`SHIP` · Week 1 · Foundations

Your week one work, public, with a README a stranger can actually follow.

- Cleaning a week of commits into something readable
- Writing a README that explains what and why, not just how to run it
- Recording a short demo of the thing working
- Posting your build publicly and handling the first questions
- Reading your own code from a stranger's point of view

**The flow:** a week of commits (messy, yours alone) → a README (what it does, and why) → a demo (60 seconds of it working) → public (LinkedIn, X, the repo)

### Day 07: Research Deep Dive & Systems Reflection

`READ` · Week 1 · Foundations

Read a paper properly, then map its claims onto the thing you just built.

- Reading a paper without reading every word: abstract, figures, conclusion
- Mapping the paper's claims onto what you built this week
- Writing a one-page reflection on what broke and why
- Identifying the one concept from week one you still cannot explain out loud

**The flow:** the paper (abstract, figures, conclusion) → the claims (what it says is true) → your build (what you observed) → the gap (the one thing you cannot explain)

---

## Week 2: Retrieval

**Days 08 to 14 · chunking, RAG 2.0, hybrid search, tools**

### Day 08: Document Chunking & The Knapsack Budget

`BUILD` · Week 2 · Retrieval

A chunking module that fits a token budget without cutting meaning in half.

- Why chunk size decides retrieval quality more than the model does
- Fixed-size, sentence-aware, and recursive chunking strategies
- Overlap, and the tradeoff it buys you
- Counting tokens per chunk before you embed anything
- Fitting chunks into a context budget as a knapsack problem
- Keeping source and position metadata on every chunk
- Testing chunk quality before you build anything on top of it

**The flow:** document (one long file) → splitter (sentence aware) → chunks (with overlap) → budget (fits the context). Chunk size trade: 256 is tight and precise, 512 is the usual start, 1024 gives more context but less precision

### Day 09: Your First Complete RAG Pipeline

`BUILD` · Week 2 · Retrieval

Ingest through answer, wired end to end, with sources cited back.

- The two phases: index once, retrieve on every query
- Wiring ingest, chunk, embed, store into one pipeline
- Retrieve, rank, and select context for the prompt
- Prompt construction with retrieved context
- Citing sources back to the user
- Handling the case where retrieval returns nothing useful
- Measuring whether the answer actually came from your documents

**The flow:** Index phase, runs once: ingest the documents → chunk and embed → write to the store. Query phase, every search: embed the question → retrieve top chunks → build the prompt (this is where latency lives)

### Day 10: Hybrid Search & Reciprocal Rank Fusion

`LIVE` · Week 2 · Retrieval

Vector and keyword search in parallel, merged with RRF, top 5 returned.

- Where vector search fails: error codes, part numbers, exact IDs
- Where keyword search fails: meaning without shared words
- Running both searches in parallel
- Why adding similarity scores together does not work
- Reciprocal Rank Fusion, and the smoothing constant k
- How k prevents top ranks from dominating the fused score
- Tuning the fusion and measuring whether it helped

**The flow:** vector search (great at meaning, blind to exact strings) + keyword search (great at exact strings, blind to meaning) → RRF: score = sum( 1 / (k + rank) ), k = 60

### Day 11: Structured Outputs & JSON Extraction

`BUILD` · Week 2 · Retrieval

An extraction service that returns valid, schema-conformant JSON every time.

- Why respond in JSON in a prompt is not enough
- Defining a schema before you write the prompt
- Native structured output modes versus prompt-only extraction
- Validating the response and what to do when it fails
- Retry strategy on malformed output
- Handling missing fields and nulls without crashing downstream
- Extracting from messy, real-world text rather than clean examples

**The flow:** model output (a string, hopefully JSON) → validate against your schema → accept (hand it downstream) or retry (with the error fed back in). Attempt 1 invalid, attempt 2 invalid, attempt 3 raise

### Day 12: Tool Calling & Function Execution

`LIVE` · Week 2 · Retrieval

An AI Math Assistant with real tools, parsed, executed locally, and verified.

- Native function calling versus asking for code in plain text
- Describing a tool so the model knows when to call it
- Parsing a tool-call request out of the response
- Executing locally and feeding the result back
- Preventing dangerous calls and malicious parameters
- Handling an exception thrown mid-execution
- Building a registry so adding a tool is one decorator

**The flow:** the model (asks for a tool) → registry (looks it up) → execute (runs locally) → result (fed back in). The model answers again, this time with a real number. Adding a tool: @register_tool def calculate(expression)

### Day 13: Weekly Ship Day: Production RAG 2.0 Showcase

`SHIP` · Week 2 · Retrieval

Your hybrid-search RAG system, public, with an honest note on where it fails.

- Packaging a multi-part pipeline so someone else can run it
- Writing up a failure mode instead of hiding it
- Demoing retrieval quality, not just a working screenshot
- Handling the first real question about your architecture choices

**The flow:** the pipeline (four moving parts) → it runs elsewhere (someone else's machine) → the failure note (where it still breaks) → public (with the retrieval numbers)

### Day 14: Research Deep Dive & RAG Failure Modes

`READ` · Week 2 · Retrieval

Name the failures you already hit this week, then find the fix for each one.

- Reading the literature on retrieval failure
- Naming the failure modes you have already hit this week
- Lost-in-the-middle, chunk boundary loss, and retrieval drift
- Mapping each failure to a fix you could implement

**The flow:** three failures you have already seen: lost in the middle (context ignored), chunk boundary (the answer got cut), retrieval drift (wrong docs, right shape). Underneath all three: retrieval failed and the answer looked confident anyway

---

## Week 3: Agents

**Days 15 to 21 · ReAct loops, sandboxes, guardrails, evals**

### Day 15: The ReAct Agent Loop From Scratch

`BUILD` · Week 3 · Agents

A ReAct loop written by hand, with no agent framework anywhere near it.

- Reason, act, observe, as three steps in one loop
- Writing the loop yourself instead of importing one
- Parsing the model's chosen action from its output
- Feeding an observation back into the next turn
- Stopping conditions and max-iteration caps
- Tracking token spend across a multi-turn loop
- Debugging an agent that loops without converging

**The flow:** reason (what should I do next) → act (call the tool) → observe (read the result), repeated until it finishes or your cap stops it. max_iterations = 5, because an agent with no cap is a bill with no cap

### Day 16: Isolated Code Execution Sandboxing

`BUILD` · Week 3 · Agents

Run model-generated code without giving it your machine.

- Why running generated code directly is not an option
- Isolation approaches and their tradeoffs
- Resource limits: time, memory, and output size
- Blocking filesystem and network access from inside the sandbox
- Capturing stdout, stderr, and exceptions cleanly
- Returning a failed execution to the model in a usable form
- Testing your sandbox by trying to break it

**The flow:** no sandbox: your filesystem, your network, and your credentials are all reachable from one bad generation. Sandboxed: time limit, memory limit, no network, no disk, so it fails safely and reports back cleanly

### Day 17: Multi-Agent Collaboration: Planner & Reviewer

`LIVE` · Week 3 · Agents

A Coder drafts, a Reviewer critiques, and it loops until approved or capped at 3.

- When multi-agent genuinely beats one well-prompted model
- Splitting a task into generator and evaluator roles
- Writing review criteria the reviewer can actually apply
- The revision loop, and tracking revision history
- Stopping on APPROVED or after a fixed attempt cap
- Preventing multi-agent loops from blowing up token cost
- Hierarchical versus choreographed state-machine designs

**The flow:** coder (drafts a solution) → reviewer (critiques it) → revision or APPROVED. Maximum three rounds, then it ships whatever it has: round 1 changes, round 2 changes, round 3 APPROVED

### Day 18: Automated Evaluation Engineering & Golden Sets

`BUILD` · Week 3 · Agents

An eval harness that tells you whether a change made things better or worse.

- Why it looks better is not a measurement
- Building a golden set from real failures, not synthetic examples
- Exact match, fuzzy match, and LLM-as-judge scoring
- Running evals in CI so regressions are caught before shipping
- Reading an eval result and deciding what to change
- Eval cost, and keeping the loop fast enough to actually use
- Versioning your golden set as your product changes

**The flow:** golden set (from real failures) → run (your pipeline) → score (match or judge) → compare (better or worse). Example: before 62/100, after 71/100, ship it

### Day 19: Security, Guardrails & Prompt Injection Defense

`LIVE` · Week 3 · Agents

Scan the input, mask the PII, and check the output before it leaves.

- Direct versus indirect prompt injection
- How an AI reading support emails gets exploited
- Why client-side validation does not protect an LLM
- Scanning input for known injection patterns
- Masking PII: emails, card numbers, phone numbers
- Checking outbound responses for leaked system instructions
- The Dual-LLM Guardrail pattern and its latency cost

**The flow:** incoming text (user, or a document) → guardrail (injection + PII scan) → pass (masked, then to the model) or block (and log what tripped it). mask_pii_entities(text) handles emails, 16-digit cards, and phone numbers

### Day 20: Weekly Ship Day: Agentic System Showcase

`SHIP` · Week 3 · Agents

Your agent, public, with the guardrail layer and what it still cannot stop.

- Demoing an agent without editing out the failed runs
- Documenting your security assumptions explicitly
- Showing the eval numbers alongside the demo

**The flow:** the agent (running, unedited) → the guardrail (and what it catches) → the eval score (not a vibe) → what it cannot stop (stated openly)

### Day 21: Research Deep Dive & The ReAct Paradigm

`READ` · Week 3 · Agents

Read the paper that named the loop, then see where your version diverged.

- The ReAct paper, and how it differs from what you built
- Where the original paradigm has been superseded
- Mapping your agent loop against the paper's loop
- Naming which of your design choices were yours, not the paper's

**The flow:** the paper: reason, act, observe, with no tool registry and no cost tracking (2022, and still the base). What you built: the same loop plus a tool registry and token tracking. Those additions are yours, so name them

---

## Week 4: Production

**Days 22 to 30 · serving, FDE skills, capstone system design**

### Day 22: High-Throughput APIs with FastAPI

`BUILD` · Week 4 · Production

Wrap your pipeline in an API that holds up when more than one person calls it.

- Wrapping a pipeline in an HTTP API
- Async endpoints and why blocking calls kill throughput
- Request and response models with Pydantic
- Background tasks for anything slow
- Error handling that returns a usable response, not a stack trace
- Health checks and readiness endpoints
- Load testing your own endpoint before anyone else does

**The flow:** request (validated by Pydantic) → async endpoint (never blocks) → your pipeline (retrieval + model) → response (typed, not a traceback). GET /health returns 200, because something has to check it

### Day 23: Hardware Capacity Sizing & Memory Calculations

`BUILD` · Week 4 · Production

Turn a user count into a hardware number and a monthly bill.

- Model weights, precision, and memory footprint
- KV cache, and why it grows with context length
- Batch size versus latency, as a tradeoff you choose
- Calculating required QPS from user counts
- Estimating daily token consumption
- Turning tokens into monthly hosting cost
- Sizing for peak, not average

**The flow:** 10,000 users (the only input you are given) → required QPS (sized for peak) → tokens per day (prompt plus completion) → monthly cost (the number they ask for)

### Day 24: Real-Time Streaming with Server-Sent Events

`LIVE` · Week 4 · Production

Stream tokens as they arrive, so the first word lands in milliseconds.

- Server-Sent Events versus WebSockets for AI chat
- Time-to-First-Token and perceived latency
- Async generators and yielding tokens as they arrive
- SSE message format and client handling
- Closing an upstream generation when the user leaves
- Buffering, backpressure, and dropped connections
- Testing a stream without a frontend

**The flow:** the model (generating) → async generator (yields each token) → SSE stream (data: ...) → the browser (text appears). Each message looks like data: {"token": "Hello"}, and the user stops waiting

### Day 25: The Forward Deployed Engineer: Client Discovery & Scoping

`BUILD` · Week 4 · Production

Find the real problem, scope something that can ship, and price it.

- What a Forward Deployed Engineer actually does
- Running a discovery call that finds the real problem
- Separating the stated problem from the underlying one
- Scoping a build that can ship, not one that sounds impressive
- Writing assumptions down before you commit to them
- Pricing a scope and defending it
- Knowing when to say a project should not be built

**The flow:** discovery call (questions, not a demo) → the real problem (under the stated one) → a scope (that can actually ship) → a price (you can defend). Stated: "we need an AI chatbot". Real: "support tickets take 3 days"

### Day 26: Safe Enterprise Database Integration: Text-to-SQL

`BUILD` · Week 4 · Production

Query a real database without letting the model damage it.

- Giving the model a schema without giving it the database
- Read-only connections and least-privilege access
- Validating generated SQL before executing it
- Blocking writes, drops, and anything destructive
- Query timeouts and row limits
- Returning results in a form the model can summarise
- Handling a query that returns nothing, or too much

**The flow:** a question (in plain English) → SQL validator (read-only, no DROP) → execute (with a timeout and row cap) or reject (and say why). SELECT allowed. INSERT, UPDATE, DELETE, DROP blocked

### Day 27: Weekly Ship Day: Containerizing & Deploying Your AI Service

`SHIP` · Week 4 · Production

Your service, containerized and live at a URL a stranger can hit.

- Writing a Dockerfile for a Python AI service
- Environment variables and secrets in a container
- Image size, and why it matters for deploy time
- Deploying and getting a public URL
- Basic logging and knowing when it breaks

**The flow:** your code (runs on your machine) → Dockerfile (runs anywhere) → image (built and tagged) → a URL (anyone can hit it). docker build -t my-service . && docker run -p 8000:8000 my-service

### Day 28: Capstone AI System Design: The 10,000-User Blueprint

`LIVE` · Week 4 · Production

A full architecture blueprint and a 2-page design doc for 10,000 users.

- Designing for 10,000 concurrent users without hitting rate limits
- Where the caching layer goes, and the invalidation strategy
- Read and write path separation
- Capacity estimation from user counts to cluster size
- PagedAttention and how pagination prevents GPU fragmentation
- Writing a 2-page design doc a stakeholder will actually read
- Defending your architecture choices under questioning

**The flow:** 10,000 users (the load you were handed) → cache layer (and its invalidation rule) → retrieval + model (the expensive part) → a 2-page doc (that a stakeholder reads)

### Day 29: Portfolio Assembly & Technical Case Study

`BUILD` · Week 4 · Production

Pick your strongest build and write the case study a hiring manager will skim.

- Choosing which of your 30 builds to lead with
- Writing a case study with the problem, the decision, and the tradeoff
- Showing the code that matters, hiding the code that does not
- Turning a GitHub history into a narrative
- Writing for a hiring manager who will skim it in 90 seconds

**The flow:** 30 builds (all public, all dated) → pick one (your strongest) → case study (problem, decision, tradeoff) → portfolio (skimmable in 90 seconds)

### Day 30: Grand Showcase, Portfolio Release & Graduation

`SHIP` · Week 4 · Production

Publish, post, submit, and graduate with work instead of a certificate.

- Publish your final technical case study on Substack
- Post your complete 30-day GitHub build history on LinkedIn and X
- Submit your portfolio to the Buildership Showcase
- Graduate

**The flow:** 30 days, 30 builds (every one of them public) → Substack (your case study) + LinkedIn and X (the full build history) + the Showcase (your portfolio submitted)

---

## The 8 Live Sessions

Wednesday and Friday, 7:30 PM IST, 90 minutes each.

| Session | Day | Topic |
|---|---|---|
| 1 | Day 03 | Tokenization & Character Blindness |
| 2 | Day 05 | Building an In-Memory Vector Store |
| 3 | Day 10 | Hybrid Search & Reciprocal Rank Fusion |
| 4 | Day 12 | Tool Calling & Function Execution |
| 5 | Day 17 | Multi-Agent: Planner & Reviewer |
| 6 | Day 19 | Guardrails & Prompt Injection Defense |
| 7 | Day 24 | Real-Time Streaming with SSE |
| 8 | Day 28 | Capstone: The 10,000-User Blueprint |

Day 1 was also taught live this season, outside the roadmap tags.

## Ship Days and Read Days

| Type | Day | Topic |
|---|---|---|
| SHIP | Day 06 | Ship Day: Week 1 Public Proof |
| READ | Day 07 | Read Day: Systems Reflection |
| SHIP | Day 13 | Ship Day: Production RAG 2.0 Showcase |
| READ | Day 14 | Read Day: RAG Failure Modes |
| SHIP | Day 20 | Ship Day: Agentic System Showcase |
| READ | Day 21 | Read Day: The ReAct Paradigm |
| SHIP | Day 27 | Ship Day: Containerize & Deploy |
| SHIP | Day 30 | Grand Showcase & Graduation |

---

## How the Days Connect

The builds stack. Where the roadmap says it outright, the link is stated. Where the topics line up, it is the natural reading.

- Day 01 starts the token tracking. Day 15 tracks token spend across a loop, and Day 21 names token tracking as one of your additions to the ReAct paper.
- Day 04 (embeddings) feeds Day 05 (vector store), which loads back from disk without re-embedding.
- Day 08 (chunking) joins embeddings and the store in the complete RAG pipeline on Day 09.
- Day 10 (hybrid search) upgrades that pipeline, and Day 13 ships it as Production RAG 2.0.
- Day 12 builds the tool registry, and Day 21 names that registry as part of the agent you built.
- Day 18 (evals) and Day 19 (guardrails) are the two layers the Day 20 agent showcase puts on screen.
- Day 22 (FastAPI) wraps your pipeline, and Day 27 containerizes and deploys the service.
- Day 23 (sizing) and Day 28 (the 10,000-user blueprint) both turn a user count into capacity and cost.
- Day 29 and Day 30 turn the 30 dated builds into one case study and one portfolio.

---

## What You Leave With

Not a certificate. A dated, public body of work with your name on it.

| After 30 days | |
|---|---|
| 30 | builds on GitHub, dated and public |
| 8 | live build sessions, 90 minutes each |
| 90 | interview questions answered |
| 30 | technical problems solved |
| 30 | system design patterns |
| 30 | AI-DSA topics |
| 1 | published case study |
| 1 | portfolio, not a certificate |

On graduation day: your 30-day history (every commit dated and public), your case study on Substack, the full build history on LinkedIn and X, and your portfolio submitted to the Showcase.