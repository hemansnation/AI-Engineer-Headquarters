from data.samples import EMAILS
from guard.pipeline import build_assistant
assistant = build_assistant()
for email in EMAILS:
    result = assistant.handle(email['body'], source='email')
    got = 'pass' if result['allowed'] else 'block'
    print(email['id'], 'expected', email['expected'], 'got', got, 'stage', result['stage'], result['reasons'])
    print('   masked', result['pii_masked'], 'answer', result['answer'][:80])
