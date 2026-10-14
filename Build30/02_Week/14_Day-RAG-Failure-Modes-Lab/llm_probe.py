import os
import time
from dotenv import load_dotenv
from google import genai
from corpus import DOCS
from golden_set import GOLDEN
load_dotenv()
MODEL = os.getenv('GEMINI_MODEL')
if not MODEL:
    raise SystemExit('Set GEMINI_MODEL in .env to the free model you used on Day 1')
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
def call_llm(prompt):
    return client.models.generate_content(model=MODEL, contents=prompt).text or ''
def build_context(gold_text, distractors, position):
    texts = list(distractors)
    texts.insert(position, gold_text)
    return '\n\n'.join(f'Document {n + 1}: {t}' for n, t in enumerate(texts))
def main():
    by_id = {d['doc_id']: d for d in DOCS}
    positions = [0, 3, 5]
    score = {p: 0 for p in positions}
    asked = [g for g in GOLDEN if g['gold_doc']]
    for g in asked:
        gold_text = by_id[g['gold_doc']]['text']
        others = [d['text'] for d in DOCS if d['doc_id'] != g['gold_doc'] and d['current']][:5]
        for p in positions:
            context = build_context(gold_text, others, p)
            prompt = 'Use only these documents.\n\n' + context + '\n\nQuestion: ' + g['question'] + '\nAnswer in one short sentence.'
            answer = call_llm(prompt)
            score[p] += g['key'].lower() in answer.lower()
            time.sleep(2)
    for p in positions:
        print('gold document at position', p, 'correct', score[p], 'of', len(asked))
if __name__ == '__main__':
    main()
