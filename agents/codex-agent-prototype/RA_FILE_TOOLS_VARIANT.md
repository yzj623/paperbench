# Codex with RA file tools

This variant exposes the RA main-agent `read_file_chunk`, `search_file`, and
`write_file` contracts through Codex app-server's existing `dynamicTools` API.
The original image `codex-paperbench:latest` remains available. The variant is
built as `codex-paperbench-ra-file-tools:latest`.

Build or rebuild the variant:

```bash
./build_codex_image.sh --image codex-paperbench-ra-file-tools:latest --jobs 1
```

Launch an experiment by passing the normal launcher options to the variant
wrapper. For example:

```bash
./run_ra_file_tools.sh --paper-id all-in-one --plan /path/to/plan.json --no-checker
```

The main agent receives the three RA tools and is instructed to use them for
reading and writing. Codex's native command tools remain available for
lightweight checks. Planner and checker tools are unchanged. The variant is
opt-in: the original launcher does not add the RA file tools.
