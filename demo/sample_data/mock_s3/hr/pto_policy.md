# PTO Policy

## Employee Questions

Internal HR questions should be classified as policy questions. The demo keeps
the policy content small, but still returns citations when evidence is enough.

Employees may ask "PTO 是什么意思？" or "How should HR policy questions be
retrieved?" The agent should route these questions to policy retrieval and cite
HR documents. This sample is intentionally separate from engineering documents
so metadata filters by department are easy to demonstrate.

## Acronyms

PTO means paid time off. HR policy documents are good examples for custom
dictionary and synonym demos.

## Synonym Dictionary

The workshop can map `PTO`, `paid time off`, `假期`, and `休假` as synonyms.
Milvus custom dictionary and synonym dictionary features are useful because HR
questions often mix English acronyms with Chinese wording.

## Citation Behavior

The assistant should not answer HR policy questions from memory alone. It
should retrieve a policy chunk, grade evidence, and cite the relevant
`chunk_id`. If evidence is weak, it should ask for a more specific policy name.
