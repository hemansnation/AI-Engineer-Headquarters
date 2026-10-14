def reciprocal_rank_fusion(rankings, k=60, top_n=5, weights=None):
    if weights is None:
        weights = [1.0] * len(rankings)
    fused = {}
    for weight, ranking in zip(weights, rankings):
        for position, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + position)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:top_n]
def naive_score_sum(score_lists, top_n=5):
    fused = {}
    for scores in score_lists:
        for doc_id, score in scores:
            fused[doc_id] = fused.get(doc_id, 0.0) + score
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:top_n]
