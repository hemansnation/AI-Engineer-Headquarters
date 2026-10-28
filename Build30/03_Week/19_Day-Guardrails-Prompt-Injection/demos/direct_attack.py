from data.samples import DIRECT_ATTACKS
from guard.pipeline import build_assistant
assistant = build_assistant()
for text in DIRECT_ATTACKS:
    result = assistant.handle(text, source='user')
    print(result['allowed'], result['stage'], result['reasons'], result['timings'])
    print('   ', result['answer'])
