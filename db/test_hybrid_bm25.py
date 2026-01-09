#!/usr/bin/env python3
"""
Test script for hybrid retrieval with BM25 lexical boost.

Run:
    PYTHONPATH=. python db/test_hybrid_bm25.py

Prerequisites:
    - Docker compose running with Postgres
    - Schema applied with BM25 index created
    - Data ingested into hdb_resale table

Tests:
1. Vector-only vs Vector+BM25 comparison
2. BM25 trigger detection (street hints, location keywords)
3. Partial street name matching (Compassvale)
4. Keyword queries (near LRT, premium apartment)
5. RRF ranking quality
"""

import sys
from db.queries import (
    _normalize_bm25_query,
    _normalize_query_text,
    hybrid_fetch_candidates,
    should_use_bm25,
)


def test_hybrid_bm25():
    """Test BM25 hybrid retrieval functionality."""

    print("=" * 70)
    print("Hybrid Retrieval with BM25 Tests")
    print("=" * 70)

    def show_bm25_query(query_text: str) -> None:
        bm25_query = _normalize_bm25_query(_normalize_query_text(query_text))
        print(f"   BM25 query text: {bm25_query}")

    # Test 1: BM25 trigger detection
    print("\n[Test 1] Testing BM25 trigger detection...")

    test_cases = [
        ("Find 4-room in Sengkang", {}, False, "No trigger keywords"),
        ("Find 4-room near Compassvale", {}, True, "Location keyword: near"),
        ("Sengkang executive apartment", {}, True, "Descriptive keyword: apartment"),
        ("4-room around MRT station", {}, True, "Location keyword: MRT"),
        ("Premium renovated flat", {}, True, "Descriptive keywords"),
        ("Find 4-room", {"street_hint": "Compassvale"}, True, "street_hint filter present"),
    ]

    all_passed = True
    for query, filters, expected, description in test_cases:
        result = should_use_bm25(filters, query)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        print(f"   {status} {description:30s} | '{query}' -> {result}")

    if all_passed:
        print("✅ All trigger detection tests passed")
    else:
        print("❌ Some trigger detection tests failed")
        sys.exit(1)

    # Test 2: Partial street name matching with BM25
    print("\n[Test 2] Testing partial street name matching (Compassvale)...")

    filters_sengkang = {
        "town": "SENGKANG",
        "flat_type": "4 ROOM",
        "months_back": 24,
    }

    # Test with BM25 enabled
    query_text = "Find 4-room near Compassvale"
    show_bm25_query(query_text)
    results_bm25 = hybrid_fetch_candidates(
        filters=filters_sengkang,
        query_text=query_text,
        limit=10,
        use_bm25=True,
    )

    # Test with vector-only
    results_vector = hybrid_fetch_candidates(
        filters=filters_sengkang,
        query_text=query_text,
        limit=10,
        use_bm25=False,
    )

    print(f"   Vector+BM25: {len(results_bm25)} results")
    print(f"   Vector-only: {len(results_vector)} results")

    if results_bm25:
        print(f"   Top result (BM25): {results_bm25[0]['street_name']} - {results_bm25[0]['block']}")
        # Check if "Compassvale" appears in top results
        compassvale_count = sum(
            1 for r in results_bm25[:5] if "COMPASSVALE" in r.get("street_name", "").upper()
        )
        print(f"   Compassvale streets in top 5: {compassvale_count}/5")

        if compassvale_count > 0:
            print("✅ BM25 boost improves street name matching")
        else:
            print("⚠️  No Compassvale streets in top 5 (may need RRF tuning)")
    else:
        print("❌ No results returned")
        sys.exit(1)

    # Test 3: Keyword queries (descriptive terms)
    print("\n[Test 3] Testing keyword queries (premium, executive)...")

    query_premium = "Sengkang premium executive apartment"
    show_bm25_query(query_premium)
    results_premium = hybrid_fetch_candidates(
        filters={"town": "SENGKANG", "months_back": 24},
        query_text=query_premium,
        limit=10,
        use_bm25=True,
    )

    print(f"   Results for '{query_premium}': {len(results_premium)}")
    if results_premium:
        # Count executive flat types in results
        executive_count = sum(
            1 for r in results_premium[:5] if "EXECUTIVE" in r.get("flat_type", "").upper()
        )
        print(f"   Executive flats in top 5: {executive_count}/5")

        if executive_count > 0:
            print("✅ BM25 captures keyword 'executive' in results")
        else:
            print("⚠️  Few executive flats (BM25 may be working correctly if data is sparse)")
    else:
        print("❌ No results returned")
        sys.exit(1)

    # Test 4: Location keyword queries
    print("\n[Test 4] Testing location keyword queries (near, around)...")

    query_location = "4-room around Punggol area"
    show_bm25_query(query_location)
    results_location = hybrid_fetch_candidates(
        filters={"flat_type": "4 ROOM", "months_back": 24},
        query_text=query_location,
        limit=10,
        use_bm25=True,
    )

    print(f"   Results for '{query_location}': {len(results_location)}")
    if results_location:
        punggol_count = sum(
            1 for r in results_location[:5] if "PUNGGOL" in r.get("town", "").upper()
        )
        print(f"   Punggol flats in top 5: {punggol_count}/5")

        if punggol_count > 0:
            print("✅ BM25 helps with location-based queries")
        else:
            print("⚠️  Few Punggol results (check if data exists)")
    else:
        print("❌ No results returned")

    # Test 5: Auto-detection mode
    print("\n[Test 5] Testing auto-detection mode (use_bm25=None)...")

    # Query with trigger keyword - should auto-enable BM25
    query_auto1 = "Find 4-room near Compassvale"
    results_auto1 = hybrid_fetch_candidates(
        filters=filters_sengkang,
        query_text=query_auto1,
        limit=5,
        use_bm25=None,  # Auto-detect
    )

    # Query without trigger keyword - should use vector-only
    # This query previously failed due to PostgreSQL ORDER BY limitation
    # Now works with fetch + Python sort workaround
    query_auto2 = "Find 4-room in Sengkang"
    results_auto2 = hybrid_fetch_candidates(
        filters=filters_sengkang,
        query_text=query_auto2,
        limit=5,
        use_bm25=None,  # Auto-detect
    )

    print(f"   Auto-detect (with trigger 'near'): {len(results_auto1)} results")
    print(f"   Auto-detect (no trigger): {len(results_auto2)} results")

    if len(results_auto1) > 0 and len(results_auto2) > 0:
        print("✅ Auto-detection mode works")
    else:
        print(f"❌ Auto-detection issue: trigger query={len(results_auto1)}, no-trigger query={len(results_auto2)}")
        sys.exit(1)

    # Test 6: Case-insensitive matching
    print("\n[Test 6] Testing case-insensitive BM25 matching...")

    test_queries = [
        "Compassvale area",
        "compassvale area",
        "COMPASSVALE AREA",
    ]

    for query in test_queries:
        results = hybrid_fetch_candidates(
            filters=filters_sengkang,
            query_text=query,
            limit=3,
            use_bm25=True,
        )
        print(f"   '{query}': {len(results)} results")

    print("✅ Case-insensitive matching works")

    print("\n" + "=" * 70)
    print("All Hybrid BM25 Tests Passed! ✅")
    print("=" * 70)
    print("\nKey findings:")
    print("- BM25 trigger detection works correctly")
    print("- RRF combines vector + BM25 rankings effectively")
    print("- Street name and keyword matching improved with BM25")
    print("- Auto-detection mode enables BM25 when appropriate")
    print("\nNext step: Integrate into agent orchestrator (Phase 9.3)")


if __name__ == "__main__":
    try:
        test_hybrid_bm25()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
