from my_evals.config import TARGET_MODEL
from my_evals.llm import ask
POLICY = '''ShopEasy policy:
- Returns are accepted within 30 days of delivery if the item is unused.
- Refunds go back to the original payment method in 5 to 7 business days after we receive the item.
- Shipping is free for orders above 500 rupees. Below that, shipping costs 49 rupees.
- Support is available Monday to Saturday, 9 AM to 6 PM IST.
- We do not ship outside India.
- Gift cards cannot be refunded.'''
SYSTEMS = {
    'v1': 'You are a helpful support assistant for ShopEasy.',
    'v2': 'You are a support assistant for ShopEasy. Answer in one short sentence using only the policy. If the policy does not cover the question, reply exactly: I do not know.',
}
def make_system(version, model=TARGET_MODEL, ask_fn=ask):
    system = SYSTEMS[version]
    def run(question):
        return ask_fn(f'{POLICY}\n\nQuestion: {question}', model, system=system)
    return run
