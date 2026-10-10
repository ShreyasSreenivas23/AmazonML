"""Prompt construction for the LLM reranker (no GPU, no model download)."""
import pandas as pd

from src.er_llmrank import competitors_of, prompts_for


def test_prompts_show_competitors_excluding_own_s1():
    chunks = pd.DataFrame({"s1_id": ["A", "B", "C", "A"], "cand_id": ["r1", "r1", "r1", "r2"],
                           "prob": [0.4, 0.6, 0.1, 0.9]})
    comp = competitors_of(chunks)
    assert comp["r1"] == ["B", "A", "C"]
    left = pd.DataFrame({"business_name": ["Best Technology", "Best Technology LLP", "Best Tech"],
                         "business_address": ["Thane", "", "Pune"], "country": ["India"] * 3}, index=["A", "B", "C"])
    right = pd.DataFrame({"business_name": ["BEST TECHNOLOGY", "x"], "business_address": ["", "y"]}, index=["r1", "r2"])
    p = prompts_for(pd.DataFrame({"s1_id": ["A"], "cand_id": ["r1"]}), comp, left, right)[0]
    assert "Business A (India): Best Technology | Thane" in p and "Record B: BEST TECHNOLOGY | (no address)" in p
    assert "Best Technology LLP | (no address)" in p and "Best Tech | Pune" in p
    assert p.count("  - ") == 2 and p.rstrip().endswith("Answer:")      # own S1 (A) is not listed as a competitor
