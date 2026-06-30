"""
Model accuracy checker for VESPER cross-modal satellite image retrieval.
Reads metrics.json and prints a structured accuracy report.
"""
import json
import os
import sys


def load_metrics():
    candidates = [
        os.path.join(os.path.dirname(__file__), "metrics.json"),
        "backend/metrics.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    print("ERROR: metrics.json not found. Run eval.py first.")
    sys.exit(1)


def pct(v):
    return f"{v * 100:.2f}%"


def main():
    m = load_metrics()
    print("=" * 64)
    print("  VESPER Model Accuracy Report")
    print("=" * 64)

    # --- Same-modal ---
    print("\n1. SAME-MODAL RETRIEVAL (category-level)")
    print("-" * 50)
    for key, label in [("s1_to_s1", "S1→S1 (SAR→SAR)"), ("s2_to_s2", "S2→S2 (Optical→Optical)")]:
        d = m["same_modal"][key]
        print(f"\n  {label}")
        print(f"    Precision@5:  {pct(d['precision_at_5'])}")
        print(f"    Precision@10: {pct(d['precision_at_10'])}")
        print(f"    Hit Rate@5:   {pct(d['hit_rate_at_5'])}")
        print(f"    Hit Rate@10:  {pct(d['hit_rate_at_10'])}")
        print(f"    MAP:          {d['map']:.4f}")

    # --- Cross-modal ---
    print("\n2. CROSS-MODAL RETRIEVAL (after alignment)")
    print("-" * 50)
    for key, label in [("s1_to_s2", "S1→S2 (SAR query → Optical gallery)"),
                        ("s2_to_s1", "S2→S1 (Optical query → SAR gallery)")]:
        d = m["cross_modal"][key]
        inst = d["instance"]
        cat = d["category"]
        print(f"\n  {label}")
        print(f"    Instance-level (exact pair matching):")
        print(f"      MAP:        {inst['map']:.4f}")
        print(f"      F1@5:       {inst['f1_at_5']:.4f}")
        print(f"      F1@10:      {inst['f1_at_10']:.4f}")
        print(f"    Category-level (same land-cover class):")
        print(f"      Precision@5:  {pct(cat['precision_at_5'])}")
        print(f"      Precision@10: {pct(cat['precision_at_10'])}")
        print(f"      Hit Rate@5:   {pct(cat['hit_rate_at_5'])}")
        print(f"      Hit Rate@10:  {pct(cat['hit_rate_at_10'])}")
        print(f"      MAP:          {cat['map']:.4f}")

    # --- Performance ---
    print("\n3. QUERY LATENCY")
    print("-" * 50)
    print(f"    Avg query time: {m['performance']['avg_query_time_ms']:.2f} ms")

    # --- Overall verdict ---
    s2s2_p5 = m["same_modal"]["s2_to_s2"]["precision_at_5"]
    cross_cat_map = (m["cross_modal"]["s1_to_s2"]["category"]["map"]
                     + m["cross_modal"]["s2_to_s1"]["category"]["map"]) / 2
    cross_inst_map = (m["cross_modal"]["s1_to_s2"]["instance"]["map"]
                      + m["cross_modal"]["s2_to_s1"]["instance"]["map"]) / 2

    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  Same-modal S2→S2 Precision@5:       {pct(s2s2_p5)}")
    print(f"  Cross-modal category MAP (avg):      {cross_cat_map:.4f}")
    print(f"  Cross-modal instance MAP (avg):      {cross_inst_map:.4f}")

    if cross_cat_map > 0.7:
        print("\n  Verdict: Category-level retrieval is STRONG (MAP > 0.70).")
    elif cross_cat_map > 0.5:
        print("\n  Verdict: Category-level retrieval is MODERATE (MAP 0.50–0.70).")
    else:
        print("\n  Verdict: Category-level retrieval is WEAK (MAP < 0.50).")

    if cross_inst_map > 0.5:
        print("  Instance-level matching is MODERATE — the alignment model")
        print("  can often find the exact SAR↔Optical pair.")
    elif cross_inst_map > 0.3:
        print("  Instance-level matching is FAIR — exact pair retrieval works")
        print("  some of the time but has room for improvement.")
    else:
        print("  Instance-level matching is WEAK — exact pair retrieval is")
        print("  unreliable; consider a stronger alignment model (e.g. MLP).")

    print()


if __name__ == "__main__":
    main()
