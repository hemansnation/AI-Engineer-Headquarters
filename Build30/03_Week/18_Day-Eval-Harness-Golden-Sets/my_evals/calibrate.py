def agreement(judge_labels, human_labels):
    if len(judge_labels) != len(human_labels) or not judge_labels:
        raise ValueError('need two equal non-empty label lists')
    same = sum(j == h for j, h in zip(judge_labels, human_labels))
    return same / len(judge_labels)
