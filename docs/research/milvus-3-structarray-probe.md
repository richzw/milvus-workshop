# Milvus 3.0 StructArray Capability Probe

Status: passed for Phase 9 implementation · Date: 2026-08-24

## Scope and environment

The disposable probe ran against `milvusdb/milvus:v3.0.0` with PyMilvus 3.0.1. It created a synthetic two-parent collection, one `ARRAY<STRUCT>` field, separate HNSW indexes for `MAX_SIM_COSINE` and `COSINE`, and scalar subfields that expose a deliberate parallel-array false positive.

The probe used only synthetic vectors and metadata. It did not measure production-scale capacity, long-document quality, index bytes or build resource peaks; those remain Phase 9 evaluation outputs.

## Observations

1. StructArray schema creation, natural nested insertion, bracket-path index creation and logical nested output round-tripped successfully.
2. `MATCH_ANY` evaluated multiple scalar conditions at the same element offset and rejected the constructed cross-offset false positive.
3. `element_filter` limited element-level ANN participation. Element hits returned a top-level integer `offset`; the requested `passages` field was reconstructed as a logical list.
4. One parent produced multiple element hits with distinct offsets. Same-StructArray element-only hybrid search preserved those element identities.
5. An `EmbeddingList` query with `MAX_SIM_COSINE` returned parent identity without an offset. The exact synthetic match produced the expected MaxSim total of `2.0`.
6. Mixing element search with EmbeddingList in one native hybrid search collapsed the result to parent identity and removed the citeable offset. This confirms that citation-producing fusion must happen after separate searches using stable `chunk_id`.
7. Passing a parent filter only to the outer `hybrid_search` call did not constrain all element results in this server/SDK combination. Putting the permission/version predicate in every `AnnSearchRequest.expr` did. Phase 9 therefore requires per-request parent-filter pushdown plus authoritative `kb_chunks` revalidation before a result becomes evidence.

## Locked implementation choices

- Native baseline: Milvus 3.0.0 and PyMilvus 3.0.1 API shapes.
- Parent shortlist: TokenANN, `emb_list_rerank=true`, provisional `retrieval_ann_ratio=3.0`.
- Element evidence: `COSINE`, mandatory non-negative offset, bounded identity fields, then exact flat-chunk rehydration.
- Two separate vector subfields are required because their metric families differ.
- Native mixed-granularity hybrid search is excluded from citation production.
- Non-disabled runtime activation is fingerprint- and count-gated; any mismatch fails closed.

## Unsupported or deferred combinations

- Nested Struct/Array/JSON and sparse/function subfields inside StructArray.
- Parent-only or collapsed hits as citation evidence.
- LEMUR training lifecycle and MUVERA promotion without a measured TokenANN budget violation.
- Arbitrary user-authored element expressions. The implementation accepts only registered AND predicates over scalar allow-list fields.
- Production capacity claims. The synthetic probe does not establish maximum parent count, list-length distribution, latency budget or index-size budget.

## Sources

- [Array of Structs](https://milvus.io/docs/array-of-structs.md)
- [Create a StructArray field](https://milvus.io/docs/create-structarray-field.md)
- [Index StructArray fields](https://milvus.io/docs/index-structarray-fields.md)
- [Basic vector search with StructArray](https://milvus.io/docs/basic-vector-search-with-structarray.md)
- [Filtered search with StructArray](https://milvus.io/docs/filtered-search-with-structarray.md)
- [Choose an EmbeddingList search strategy](https://milvus.io/docs/choose-an-embeddinglist-search-strategy.md)
- [Hybrid search with StructArray](https://milvus.io/docs/hybrid-search-with-structarray.md)
