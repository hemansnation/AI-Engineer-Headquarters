from my_evals.config import GOLDEN
from my_evals.golden import Case, save_golden
from my_evals.scorers import SCORERS
from my_evals.golden import validate
CASES = [
    Case('ret-window', 'How many days do I have to return an item?', '30 days', 'contains', ['returns'], 'ticket-sample-01'),
    Case('refund-time', 'How long does a refund take?', '5|7|business days', 'contains', ['refunds'], 'ticket-sample-02'),
    Case('ship-free', 'What is the minimum order for free shipping?', '500', 'contains', ['shipping'], 'ticket-sample-03'),
    Case('ship-fee', 'What is the shipping fee for a small order?', 'Shipping costs 49 rupees.', 'edit', ['shipping'], 'ticket-sample-04'),
    Case('support-hours', 'When is support available?', 'saturday|9', 'contains', ['support'], 'ticket-sample-05'),
    Case('ship-abroad', 'Can you ship to the USA?', 'No. We only ship within India.', 'judge', ['shipping'], 'ticket-sample-06'),
    Case('gift-card', 'Can I get a refund on a gift card?', 'No. Gift cards cannot be refunded.', 'judge', ['refunds'], 'ticket-sample-07', True),
    Case('price-match', 'Do you price match Amazon?', 'I do not know.', 'judge', ['out_of_policy'], 'ticket-sample-08', True),
    Case('bitcoin', 'Can I pay in bitcoin?', 'I do not know.', 'judge', ['out_of_policy'], 'ticket-sample-09', True),
    Case('refund-where', 'Where does my refund go?', 'Your refund goes back to the original payment method.', 'embed', ['refunds'], 'ticket-sample-10'),
    Case('used-item', 'Can I return a used item?', 'No. Returns are only accepted for unused items.', 'judge', ['returns'], 'ticket-sample-11'),
    Case('days-only', 'What is the return window? Answer with only the number of days.', '30 days', 'exact', ['returns', 'format'], 'ticket-sample-12'),
]
if __name__ == '__main__':
    validate(CASES, SCORERS)
    save_golden(GOLDEN, CASES)
    print(f'wrote {len(CASES)} cases to {GOLDEN}')
