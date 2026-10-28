from claims import unsupported, mask, normalize, PLACEHOLDER
SOURCES = ['Reply rate went from 2.30% to 8.7% in 14 days across 1,250 emails.']
def test_normalize_keeps_value_not_format():
    assert normalize('2.30') == '2.3'
    assert normalize('1,250') == '1250'
    assert normalize('09') == '9'
def test_supported_numbers_pass():
    assert unsupported('I moved it from 2.3% to 8.7% in 14 days.', SOURCES) == []
def test_invented_number_is_caught():
    assert unsupported('Latency dropped to 400 ms and replies hit 8.7%.', SOURCES) == ['400']
def test_mask_replaces_only_bad_numbers():
    out = mask('Latency dropped to 400 ms and replies hit 8.7%.', ['400'])
    assert PLACEHOLDER in out and '8.7%' in out and '400' not in out
