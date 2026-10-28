from llm import clean
def test_clean_removes_thinking_block():
    assert clean('<think>planning the answer</think>\nFinal text.') == 'Final text.'
def test_clean_leaves_normal_text_alone():
    assert clean('Plain answer.') == 'Plain answer.'
