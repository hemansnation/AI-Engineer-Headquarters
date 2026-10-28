import random
import uuid
from locust import HttpUser, between, task
QUESTIONS = [
    'What does a semaphore do?',
    'Why does time.sleep freeze an async endpoint?',
    'What does status 429 mean?',
    'What is p95 latency?',
    'How many numbers does all-MiniLM-L6-v2 give per sentence?',
    'What is the difference between health and readiness?',
]
class AskUser(HttpUser):
    wait_time = between(0.5, 1.5)
    def on_start(self):
        self.key = f'user-{uuid.uuid4().hex[:8]}'
    @task(10)
    def ask(self):
        self.client.post(
            '/ask',
            json={'question': random.choice(QUESTIONS)},
            headers={'x-api-key': self.key},
        )
    @task(1)
    def health(self):
        self.client.get('/health')
