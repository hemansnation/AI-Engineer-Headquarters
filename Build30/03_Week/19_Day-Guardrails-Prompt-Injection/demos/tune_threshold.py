from data.samples import CLEAN_PROMPTS, DIRECT_ATTACKS, EMAILS
from guard.semantic import SemanticDetector, load_phrases
detector = SemanticDetector(load_phrases('data/injection_phrases.txt'))
attacks = [e['body'] for e in EMAILS if e['expected'] == 'block'] + DIRECT_ATTACKS[:3]
clean = [e['body'] for e in EMAILS if e['expected'] == 'pass'] + CLEAN_PROMPTS + [DIRECT_ATTACKS[3]]
attack_scores = []
clean_scores = []
for text in attacks:
    result = detector.scan(text)
    attack_scores.append(result['score'])
    print('ATTACK', result['score'], text[:60])
for text in clean:
    result = detector.scan(text)
    clean_scores.append(result['score'])
    print('CLEAN ', result['score'], text[:60])
print('lowest attack score', min(attack_scores))
print('highest clean score', max(clean_scores))
print('a threshold between these two is your starting point')
