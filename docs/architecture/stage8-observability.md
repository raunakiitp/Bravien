# Bravien Stage 8: Observability & Telemetry

## 1. Structured Logging Standard

All agent execution operations emit structured metadata adhering to privacy boundaries:

### Emitted Telemetry Fields:
- `request_id`: Unique trace identifier
- `intent`: Detected user intent category
- `selected_tools`: Array of tool names executed
- `execution_duration_ms`: Total execution time
- `verification_status`: Status of output verification (`VERIFIED`, `UNVERIFIED`, `FAILED`)
- `final_status`: Final outcome (`COMPLETED`, `FAILED`, `CANCELLED`)

### Privacy & Redaction Constraints:
- NEVER log passwords, API keys, authentication tokens, or private secrets.
- NEVER log internal chain-of-thought system prompts.
- Redact long user payloads from debug log outputs.
