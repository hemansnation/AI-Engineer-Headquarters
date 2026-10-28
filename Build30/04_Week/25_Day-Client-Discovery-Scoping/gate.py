from config import MAX_PAYBACK_MONTHS
def monthly_saving(brief):
    needed = (brief.volume_per_month, brief.minutes_per_task, brief.cost_per_hour_usd, brief.automatable_share)
    if None in needed:
        return None
    hours = brief.volume_per_month * brief.automatable_share * brief.minutes_per_task / 60
    return round(hours * brief.cost_per_hour_usd, 2)
def run_gate(brief, quote_usd, run_cost_usd):
    blockers = []
    conditions = []
    if brief.data_access in ('none', 'unknown'):
        blockers.append('Nobody has confirmed you can read the data. Get read-only access confirmed first.')
    if not brief.owner:
        blockers.append('No client owner is named. Ask for one person who can say yes or no.')
    if not brief.baseline_metric:
        conditions.append('No baseline number exists. Measure it for 2 weeks before you build.')
    if not brief.success_metric:
        conditions.append('No success metric. Agree one number that decides if this worked.')
    saving = monthly_saving(brief)
    payback = None
    if saving is None:
        conditions.append('Savings cannot be calculated. Ask for volume, minutes per task, cost per hour and the automatable share.')
    else:
        net = saving - run_cost_usd
        if net <= 0:
            blockers.append('Monthly saving does not cover the monthly running cost.')
        else:
            payback = round(quote_usd / net, 1)
            if payback > MAX_PAYBACK_MONTHS:
                blockers.append(f'Payback is {payback} months, longer than {MAX_PAYBACK_MONTHS:g}. Build less or do not build.')
    if brief.budget_usd and quote_usd > brief.budget_usd:
        conditions.append(f'Quote is over budget by ${quote_usd - brief.budget_usd}. Cut scope or raise the budget.')
    verdict = 'NO GO FOR NOW' if blockers else ('GO WITH CONDITIONS' if conditions else 'GO')
    return {'verdict': verdict, 'blockers': blockers, 'conditions': conditions, 'monthly_saving_usd': saving, 'payback_months': payback}
