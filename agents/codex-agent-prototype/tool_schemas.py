"""Local contracts translated to app-server dynamic function specifications."""
from protocol.checker import CheckReport


# Local copy of the descriptions actually registered by Pi's main extension.
# Keep the shared protocol implementation unchanged; these are engine-facing text.
MAIN_PROTOCOL_DESCRIPTIONS = {
    "submit_subtask": "Submit the current unfinished subtask to a fresh, stateless Check Agent. An accepted review completes it; a rejected review reports every identified problem.",
    "skip_subtask": "Defer a genuinely impossible current subtask only after at least three valid Check Agent rejections. Earlier calls do not change its status.",
    "final_submit": "Submit only after all tasks are completed or validly skipped and all hard gates pass. If skipped tasks remain, an otherwise-valid first call requests explicit confirmation; call again to confirm.",
}


NO_CHECKER_PROTOCOL_DESCRIPTIONS = {
    "submit_subtask": "Submit the current subtask and mark it completed directly after protocol validation succeeds.",
    "skip_subtask": "Legacy compatibility tool. Its rejection-count gate cannot normally be reached when subtask submissions are accepted directly.",
    "final_submit": MAIN_PROTOCOL_DESCRIPTIONS["final_submit"],
}


def dynamic(schema):
    function = schema["function"]
    return {"type": "function", "name": function["name"],
            "description": function["description"], "inputSchema": function["parameters"]}


def ra_file_tools():
    """The main-agent file tool descriptions and schemas from RA."""
    return [
        {"type": "function", "name": "read_file_chunk",
         "description": "Read a chunk of lines from a file.",
         "inputSchema": {"type": "object", "properties": {
             "file": {"type": "string", "description": "Path to the file to read."},
             "start_line": {"type": "integer", "description": "Line number to start reading from (1-indexed).", "default": 1},
             "max_lines": {"type": "integer", "description": "Maximum number of lines to read (default: 50, max: 50).", "default": 50},
         }, "required": ["file"], "additionalProperties": False}},
        {"type": "function", "name": "search_file",
         "description": "Search for a keyword or phrase in a file and return matching lines with context.",
         "inputSchema": {"type": "object", "properties": {
             "file": {"type": "string", "description": "Path to the file to search."},
             "query": {"type": "string", "description": "Text to search for (case-insensitive)."},
             "context_lines": {"type": "integer", "description": "Number of lines of context to show before and after each match (default: 2).", "default": 2},
             "max_matches": {"type": "integer", "description": "Maximum number of matches to return per page (default: 5).", "default": 5},
             "page": {"type": "integer", "description": "Which page of results to return (1-indexed, default: 1).", "default": 1},
         }, "required": ["file", "query"], "additionalProperties": False}},
        {"type": "function", "name": "write_file",
         "description": ("Create or completely replace one UTF-8 source, configuration, script, or text "
                         "file under the task submission directory. The path is validated, while content "
                         "is treated as opaque text and is never interpreted as a filesystem path. Parent "
                         "directories are created automatically."),
         "inputSchema": {"type": "object", "properties": {
             "path": {"type": "string", "description": "A real absolute path below the task submission directory, or a path relative to that directory."},
             "content": {"type": "string", "description": "The complete UTF-8 file content to write."},
         }, "required": ["path", "content"], "additionalProperties": False}},
    ]


def role_tools(role, manager=None, *, include_ra_file_tools=False):
    if role == "planner":
        return []
    if role == "checker":
        return [{"type": "function", "name": "submit_review",
                 "description": "Submit the final subtask fidelity judgment after inspecting actual evidence. A schema-valid call ends this review. accepted=true requires no issues; accepted=false requires at least one issue.",
                 "inputSchema": CheckReport.model_json_schema()}]
    if role != "main" or manager is None:
        raise ValueError("main tools require a ResearchAgentTools instance")
    tools = [dynamic(schema) for schema in manager.tool_schemas()]
    descriptions = (MAIN_PROTOCOL_DESCRIPTIONS if manager.checker_enabled
                    else NO_CHECKER_PROTOCOL_DESCRIPTIONS)
    for tool in tools:
        tool["description"] = descriptions[tool["name"]]
    if include_ra_file_tools:
        tools.extend(ra_file_tools())
    return tools


def validate_call(tools, name, arguments):
    from jsonschema import Draft202012Validator
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    schema = next((tool["inputSchema"] for tool in tools if tool["name"] == name), None)
    if schema is None:
        raise ValueError("tool is not available to this role")
    # Do not include model-controlled values (which could contain secrets) in errors.
    if next(Draft202012Validator(schema).iter_errors(arguments), None) is not None:
        raise ValueError("arguments do not satisfy the tool schema")
