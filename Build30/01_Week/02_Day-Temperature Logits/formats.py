FORMATS = {
    'one_line': 'Answer in one short sentence. No intro, no extra words.',
    'bullets': 'Answer with exactly 3 bullet points. Each one starts with a dash. Nothing else.',
    'json': 'Return only a JSON object with the keys name and reason. No code fences. No extra text.',
}

def shape(question, style):
    return f'{question}\n\n{FORMATS[style]}'