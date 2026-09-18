import pandas as pd

from src.train import write_error_analysis


def test_error_report_is_aggregate_and_contains_no_prompts(tmp_path):
    secret_prompts = [
        "sensitive jailbreak prompt alpha",
        "sensitive benign prompt beta",
        "correctly classified prompt gamma",
        "correctly classified prompt delta",
    ]
    df = pd.DataFrame({
        "prompt": secret_prompts,
        "label": [1, 0, 1, 0],
        "pred": [0, 1, 1, 0],
        "score": [0.30, 0.75, 0.80, 0.10],
    })
    output = tmp_path / "error_analysis.md"
    rows = write_error_analysis(df, output)
    text = output.read_text()
    assert [row["count"] for row in rows] == [1, 1]
    assert "Held-out examples: 4" in text
    assert "Misclassified examples: 2" in text
    assert "False negatives" in text and "False positives" in text
    assert all(prompt not in text for prompt in secret_prompts)
