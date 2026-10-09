# Code review and verification report

Date: 8 October 2026. Input: uploaded `rag-doc-qa.zip`.
Environment: Linux x64, Python 3.12.14. No real provider keys were supplied in
the archive. No live Groq/Cohere call was made; Windows was not tested.

## Result

**45 automated tests passed**, with one upstream Chroma deprecation warning
when reading collection configuration. `pip check` reported no broken
requirements; Python compilation succeeded. Streamlit started headlessly;
its health endpoint and root page both returned HTTP 200. Streamlit AppTest
executed chat, source rendering, clear and error states. This was not a visual
browser/file-picker acceptance test.

## Findings and fixes

| Original issue | Evidence / effect | Fix |
|---|---|---|
| Custom overlap broken | Unique 2,300-character input split into 1000/1000/300 with no overlap; negative overlap accepted. | RecursiveCharacterTextSplitter, bounds validation and unique-text regression. |
| Stale chunks on re-upload | Upsert never removed pages/chunks absent from a shorter revised PDF. | Replace selected corpus; compatibility add_chunks replaces supplied document names. |
| Unstaged indexing | No coherent update boundary to protect the previous corpus on partial failures. | Stage batches, atomically switch active manifest after success; remove failed staging. |
| Shared index across sessions | A globally cached parameterless get_store returned one store for everyone. | Independent per-session workspace and store. |
| Unsafe upload path | DATA_DIR / uploaded filename trusted the filename. | In-memory parsing, normalized display basename, duplicate-name rejection. |
| Unhandled failures | Only RuntimeError during generation caught; ingestion/retrieval unguarded. | Error handling for uploads, retrieval, provider authentication, quota and generation. |
| Empty/scanned PDFs appeared successful | Zero extractable chunks could still produce success. | Reject all-empty PDFs; explain OCR limitations and skipped pages. |
| Input limits absent | No size/page/encryption checks. | 20 MB/200 pages per PDF, 10 PDFs, 2,000 chunks; encrypted-file handling. |
| Distance mismatch | README claimed cosine; code used Chroma's default L2. | Explicit cosine configuration, tested. |
| Embedding identity unchecked | Reopening could mix changed models with existing vectors. | Persist provider/model identity and reject incompatible reopening. |
| Local AI violated cloud-only constraint | sentence-transformers loaded locally; CPU was not explicitly selected. | Cloud default, optional lazy local mode with explicit CPU. |
| Search edge cases | No empty-index, blank-query or top-k guards. | Empty result, positive k validation, count cap. |
| Incomplete citations | Numbered context but unnumbered source UI; no chunk references. | Consistent source numbers, physical page, display chunk and full excerpt. |
| Empty/uncited output | Blind .strip(); no citation validation. | Empty/truncated response errors, no-context refusal, source-ID checks. |
| Document instructions | Prompt did not distinguish evidence from embedded commands. | Untrusted-excerpt instruction; no security guarantee claimed. |
| Overstated tests/documentation | Local embeddings required network/model download; README claimed prompts prevent hallucinations. | Offline doubles, explicit test scope and accurate limitations/interview answers. |
| Unbounded dependencies | Future package resolution could change behavior. | Tested direct version pins; optional local requirements separate. |

## Verification coverage

- Actual PDF extraction and preserved page numbers after blank pages.
- Corrupt, password-protected, blank and limit-exceeding PDFs.
- Real splitting, bounds, overlap and complete unique-text coverage.
- Real Chroma cosine configuration, ranking mechanics, persistence and clear.
- Same-name shorter revisions and removed-document cleanup.
- Early and later-batch embedding failures preserve the active index.
- Duplicate IDs/names, safe basenames and failed multi-file ingestion.
- Independent workspaces and missing-configuration UI behavior.
- Actual HTTP client with mocked Cohere responses: batches, query/document input
  types, missing keys, redacted 400/401/403/429/500 failures.
- Actual Groq SDK with mocked HTTP transport: payload and typed authentication error.
- Prompt construction, no-context refusal, invalid/missing citation IDs,
  empty output and token-limit response handling.
- Streamlit AppTest: empty-index chat, clear, answer/source display, rerun history
  and readable generation errors.
- Headless server health/root, compilation, dependency check and live script's
  no-network help path.

Commands (using the test environment's Python):

```text
python -m pytest -q
45 passed, 1 warning

python -m pip check
No broken requirements found.

python -m compileall -q app.py src tests verify_live.py
# exit 0
```

Offline keyword vectors only test pipeline mechanics. They do not prove actual
semantic retrieval quality. No API key, weight download, or paid call was needed
for these tests. A sample PDF is included at `tests/fixtures/sample.pdf`.

## Remaining acceptance work and limitations

1. Run `python verify_live.py --live` with your own evaluation embedding key and
   Groq key. It checks two sample facts and one absent fact using real providers.
   Then inspect answers and sources against your own PDFs.
2. Run the Windows setup/manual checklist in README. Windows install, browser
   file picker/rendering and performance on your laptop remain unverified.
3. Optional local MiniLM mode was not downloaded or executed in this review.
4. Valid source numbers do not prove factual support. Top-k can return irrelevant
   passages; there is no calibrated rejection threshold. Prompts cannot guarantee
   hallucination-free output or block every adversarial document instruction.
5. Chat history is display-only; follow-up queries need an explicit subject.
6. Browser sessions get independent disk workspaces. A new session starts empty;
   previous on-disk UI indexes are not automatically offered for reopening.
   Reopening the same path programmatically is tested. This avoids a shared global
   index without adding authentication to a local demo.
7. One writer per workspace; no authenticated public hosting, worker queue,
   crash-orphan collection cleanup or audited secure data erasure. Delete the
   index folder after stopping the app when you want all session data removed.
8. No OCR or robust table/layout reconstruction. English is the default model's
   intended language. Character limits differ from token limits; reduce chunk
   size if the provider rejects input length.
9. Direct dependencies are pinned, but not every transitive dependency across
   platforms. The upstream Chroma warning is not suppressed.

## Setup changes

Default cloud embeddings need a separate Cohere free evaluation key
(`COHERE_API_KEY`). Evaluation usage is quota-limited. Groq still generates
answers. HF is not used: its current pricing page lists no included credits for
free accounts. No billing or account upgrade was performed.

Use a fresh extraction. Keep your old .env privately and add the new configuration
from .env.example. Do not reuse an old Chroma index with a different embedding
model; re-index PDFs. The phase-oriented source layout remains, with
`embeddings.py` and `service.py` added to separate provider and UI orchestration.

## Deployment acceptance addendum — 9 October 2026

The earlier review above records the initial offline scope. Subsequent live browser testing is now complete for the bundled sample PDF.

- Public deployment: https://awanish-ask-your-pdfs.streamlit.app/
- Real Cohere cloud embeddings successfully indexed the three-page sample PDF into three chunks.
- A natural multi-part question returned 23.8% efficiency, a 25-year performance warranty, and a six-month cartridge replacement interval. Numbered citations and expanded document/page/chunk excerpts were checked against pages 1 and 2.
- An absent-fact question, "What is the retail price of the SolarX-2000 in rupees?", returned exactly: "I couldn't find this in the uploaded documents."
- Changed the default Groq model to `openai/gpt-oss-120b` after the previous model returned 404. Increased completion allowance to 4096 and reinforced numeric citation instructions. Rebooted the deployment to load the updated module.
- Reran the offline regression suite after generation changes: 45 passed, one upstream Chroma deprecation warning.

These checks exercise the deployed upload, indexing, retrieval, generation, source rendering, and absent-fact refusal paths using real providers. They are sample-based acceptance checks, not a guarantee of grounding for arbitrary documents. Windows installation, optional local MiniLM mode, OCR, load testing, and adversarial-document security remain outside the verified scope. The CLI `verify_live.py --live` was not the route used for this browser acceptance test.
