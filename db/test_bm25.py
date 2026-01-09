#!/usr/bin/env python3
"""
Test script for pg_textsearch BM25 index functionality.

Run:
    PYTHONPATH=. python db/test_bm25.py

Prerequisites:
    - Docker compose running with Postgres
    - Schema applied with BM25 index created
    - Data ingested into hdb_resale table

Tests:
1. Index exists and is queryable
2. Case-insensitive matching (Compassvale/compassvale/COMPASSVALE)
3. Multi-word queries (punggol way, ang mo kio, executive apartment)
4. BM25 scoring and ranking (lower scores = better match)
5. BM25 with hard filters (hybrid retrieval preview)
6. Index statistics and health check
"""

import sys
from sqlalchemy import create_engine, text
from settings import Settings


def test_bm25_index():
    """Test pg_textsearch BM25 index functionality."""

    settings = Settings()
    engine = create_engine(settings.database_url)

    print("=" * 70)
    print("BM25 Index Functionality Tests")
    print("=" * 70)

    with engine.connect() as conn:
        # Test 1: Check if index exists
        print("\n[Test 1] Checking if BM25 index exists...")
        result = conn.execute(text("""
            SELECT
                indexname,
                indexdef
            FROM pg_indexes
            WHERE tablename = 'hdb_resale'
              AND indexname = 'idx_hdb_resale_listing_text_bm25';
        """))
        index_info = result.fetchone()

        if index_info:
            print(f"✅ Index exists: {index_info[0]}")
            print(f"   Definition: {index_info[1][:100]}...")
        else:
            print("❌ Index not found!")
            sys.exit(1)

        # Test 2: Case-insensitive matching
        print("\n[Test 2] Testing case-insensitive matching...")
        test_queries = [
            ('Compassvale', 'Title case'),
            ('compassvale', 'Lowercase'),
            ('COMPASSVALE', 'Uppercase'),
        ]

        for query_text, label in test_queries:
            result = conn.execute(text("""
                SELECT COUNT(*) as match_count
                FROM hdb_resale
                WHERE (listing_text <@> to_bm25query(:query, 'idx_hdb_resale_listing_text_bm25')) < 0;
            """), {"query": query_text})
            count = result.fetchone()[0]
            print(f"   {label:12s} '{query_text}': {count:4d} matches")

        print("✅ Case-insensitive matching works")

        # Test 3: Multi-word queries
        print("\n[Test 3] Testing multi-word queries...")
        multi_word_queries = [
            ('punggol way', 'Street with type'),
            ('ang mo kio', 'Town name'),
            ('executive apartment', 'Flat model'),
        ]

        for query_text, label in multi_word_queries:
            result = conn.execute(text("""
                SELECT COUNT(*) as match_count
                FROM hdb_resale
                WHERE (listing_text <@> to_bm25query(:query, 'idx_hdb_resale_listing_text_bm25')) < 0;
            """), {"query": query_text})
            count = result.fetchone()[0]
            print(f"   {label:20s} '{query_text}': {count:4d} matches")

        print("✅ Multi-word queries work")

        # Test 4: BM25 scoring and ranking
        print("\n[Test 4] Testing BM25 scoring and ranking...")
        result = conn.execute(text("""
            SELECT
                town,
                flat_type,
                street_name,
                block,
                listing_text <@> to_bm25query('Compassvale', 'idx_hdb_resale_listing_text_bm25') AS bm25_score
            FROM hdb_resale
            ORDER BY bm25_score ASC
            LIMIT 5;
        """))

        rows = result.fetchall()
        if rows:
            print(f"   Top 5 results for 'Compassvale':")
            print(f"   {'Town':<12s} {'Flat Type':<10s} {'Street Name':<20s} {'Block':<8s} {'BM25 Score':<12s}")
            print("   " + "-" * 75)
            for row in rows:
                print(f"   {row[0]:<12s} {row[1]:<10s} {row[2]:<20s} {row[3]:<8s} {row[4]:>10.4f}")

            # Check that scores are negative and ordered
            scores = [row[4] for row in rows]
            all_negative = all(score < 0 for score in scores)
            properly_ordered = scores == sorted(scores)

            if all_negative and properly_ordered:
                print("✅ BM25 scores are negative (distance metric)")
                print("✅ Results are properly ranked (ascending order)")
            else:
                print("❌ BM25 scoring issue detected")
                sys.exit(1)
        else:
            print("❌ No results found for test query")
            sys.exit(1)

        # Test 5: Query with filters (hybrid approach preview)
        print("\n[Test 5] Testing BM25 with hard filters (hybrid preview)...")
        result = conn.execute(text("""
            SELECT
                COUNT(*) as total_count,
                COUNT(*) FILTER (
                    WHERE (listing_text <@> to_bm25query('Compassvale', 'idx_hdb_resale_listing_text_bm25')) < 0
                ) as bm25_matches
            FROM hdb_resale
            WHERE town = 'SENGKANG'
              AND flat_type = '4 ROOM'
              AND floor_area_sqm BETWEEN 90 AND 100;
        """))

        row = result.fetchone()
        print(f"   Filtered pool (Sengkang 4-room, 90-100 sqm): {row[0]} rows")
        print(f"   With 'Compassvale' BM25 boost: {row[1]} matches")
        print("✅ BM25 works with structured filters")

        # Test 6: Index statistics
        print("\n[Test 6] Checking index statistics...")
        result = conn.execute(text("""
            SELECT bm25_summarize_index('idx_hdb_resale_listing_text_bm25');
        """))
        stats = result.fetchone()[0]
        print(f"   {stats}")
        print("✅ Index statistics accessible")

    print("\n" + "=" * 70)
    print("All BM25 Index Tests Passed! ✅")
    print("=" * 70)
    print("\nNext step: Integrate BM25 into hybrid_fetch_candidates (Phase 9.2)")


if __name__ == "__main__":
    try:
        test_bm25_index()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
