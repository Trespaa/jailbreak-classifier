import pandas as pd

from src.data_prep import balance, clean, split_dataset


def test_clean_removes_missing_short_and_duplicate_prompts():
    duplicate = "this is a sufficiently long prompt"
    df = pd.DataFrame({
        "prompt": [None, "short", duplicate, f"  {duplicate}  ", "another sufficiently long prompt"],
        "label": [0, 0, 1, 1, 0],
    })
    result = clean(df)
    assert result["prompt"].tolist() == [duplicate, "another sufficiently long prompt"]


def test_balance_is_equal_and_deterministic():
    df = pd.DataFrame({
        "prompt": [f"regular prompt {i}" for i in range(8)] + [f"jailbreak prompt {i}" for i in range(3)],
        "label": [0] * 8 + [1] * 3,
    })
    first = balance(df)
    second = balance(df)
    assert first["label"].value_counts().to_dict() == {0: 3, 1: 3}
    pd.testing.assert_frame_equal(first, second)


def test_split_is_stratified_disjoint_and_deterministic():
    df = pd.DataFrame({
        "prompt": [f"regular prompt {i}" for i in range(100)] + [f"jailbreak prompt {i}" for i in range(100)],
        "label": [0] * 100 + [1] * 100,
    })
    train, val, test = split_dataset(df)
    assert (len(train), len(val), len(test)) == (140, 30, 30)
    assert all(split["label"].mean() == 0.5 for split in (train, val, test))
    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)
    train2, val2, test2 = split_dataset(df)
    assert train.index.tolist() == train2.index.tolist()
    assert val.index.tolist() == val2.index.tolist()
    assert test.index.tolist() == test2.index.tolist()
