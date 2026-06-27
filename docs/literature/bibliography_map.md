# Bibliography Map

Local PDF metadata is maintained for research traceability only. The PDFs are local references and must not be committed or pushed unless the user explicitly confirms license and permission.

## Local PDFs Under `../docs`

| Filename | Detected title | Authors | Topic | Useful concepts | Future modules | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `mgf_mir.pdf` | PENDING_TEXT_EXTRACTION | PENDING_TEXT_EXTRACTION | Martingale valuation, stochastic calculus, derivative pricing context from user-provided scope. | Martingale measures, risk-neutral valuation, stochastic processes. | `models/pricing/`, derivative validation docs, no-arbitrage checks. | LOCAL_REFERENCE; PENDING_PAGE_LEVEL_EXTRACTION; NOT_IMPLEMENTED_AS_CODE_YET |
| `black-scholes.pdf` | PENDING_TEXT_EXTRACTION | PENDING_TEXT_EXTRACTION | Black-Scholes / Black-Scholes-Merton reference from filename and user-provided scope. | Option pricing assumptions, call/put pricing, Greeks, implied volatility validation. | Future `models/pricing/black_scholes.py`; not implemented in Iteration 005. | LOCAL_REFERENCE; PENDING_PAGE_LEVEL_EXTRACTION; NOT_IMPLEMENTED_AS_CODE_YET |
| `udea,+10222-40504-1-CE.pdf` | PENDING_TEXT_EXTRACTION | PENDING_TEXT_EXTRACTION | User-provided derivatives/pricing research reference. | Pending extraction; likely supports pricing, stochastic calculus, or volatility discussion per user scope. | Future pricing bibliography and validation notes. | LOCAL_REFERENCE; PENDING_PAGE_LEVEL_EXTRACTION; NOT_IMPLEMENTED_AS_CODE_YET |
| `quantfinancepdflibro1.pdf` | PENDING_TEXT_EXTRACTION | PENDING_TEXT_EXTRACTION | Additional local quantitative finance book/reference detected under `../docs`. | Pending extraction; may support broad quant finance methods. | Future architecture/literature map updates. | LOCAL_REFERENCE; PENDING_PAGE_LEVEL_EXTRACTION; NOT_IMPLEMENTED_AS_CODE_YET |
| `modelo.pdf` | NOT_DETECTED_LOCALLY | NOT_DETECTED_LOCALLY | Expected by user but not found under `../docs` during Iteration 005 detection. | N/A | N/A | NOT_FOUND; PENDING_USER_FILE |

## Extraction Policy

- Do not invent DOI, page, section, title, or author metadata.
- If the PDF reader does not expose text/page metadata, keep `PENDING_PAGE_LEVEL_EXTRACTION`.
- Do not implement Black-Scholes, Heston, or other pricing modules from these PDFs in Iteration 005.
- Use the files as local bibliography references only until page-level extraction is completed.
