# Enterprise document assistant: design for 10000 users
## 1 Problem and goals
One paragraph on who uses it and what a good answer means. Then 3 goals, for example answer in under 3 seconds at p95, never show company A's data to company B, and stay inside a monthly budget.
## 2 Numbers
List every assumption with its value: active users, questions per user, tokens per question, busy window, peak factor, tokens per second per GPU, price per GPU hour. Then your results: queries per day, tokens per day, average and peak queries per second, GPUs, monthly cost with and without cache.
## 3 Architecture
Paste the blueprint diagram. Then one short paragraph for each box saying what it does and why it is there.
## 4 Decisions and tradeoffs
For each of cache, read and write split, routing, model serving and rate limiting, write the choice, the other option you rejected and the cost of your choice.
## 5 Failure modes
What happens when the cache is down, when a worker dies, when the index lags behind an upload and when one tenant floods the system. Say what the user sees in each case.
## 6 Open risks
List what you have not measured yet, and how you will measure it.
