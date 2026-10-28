import sys
from data.samples import EMAILS
from guard.pipeline import build_assistant
def show(result):
    print('allowed', result['allowed'], 'stage', result['stage'], 'reasons', result['reasons'])
    print('answer', result['answer'])
    print('timings', result['timings'])
def main():
    assistant = build_assistant()
    if sys.argv[1:] == ['--emails']:
        for email in EMAILS:
            print('---', email['id'], email['subject'])
            show(assistant.handle(email['body'], source='email'))
        return
    text = ' '.join(sys.argv[1:]) or 'How do I download my last invoice?'
    show(assistant.handle(text))
if __name__ == '__main__':
    main()
