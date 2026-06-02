"""
LLM Invoker

Unified LLM invocation utility providing:
- LLM calls with statistics tracking
- Robust JSON parsing (handles malformed JSON returned by LLMs)
- JSON extraction helpers
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.schema import BaseMessage

logger = logging.getLogger(__name__)


def _fix_unescaped_quotes_in_strings(json_str: str) -> str:
    """Fix unescaped double quotes inside JSON string values.

    LLMs (especially Qwen models) often produce:
        "reasoning": "The code used "planks.length" which failed"
    Which should be fixed to:
        "reasoning": "The code used \\"planks.length\\" which failed"

    Heuristic: after a closing quote ", the next non-whitespace character should
    be one of : , } ] or end of input. If not, that " is an unescaped quote
    inside the string and needs to be escaped.
    """
    if '"' not in json_str:
        return json_str

    result = []
    i = 0
    in_string = False

    while i < len(json_str):
        char = json_str[i]

        if char == '\\' and in_string:
            # Escape character — preserve as-is and skip the next character
            result.append(char)
            if i + 1 < len(json_str):
                i += 1
                result.append(json_str[i])
            i += 1
            continue

        if char == '"':
            if not in_string:
                # Open string
                in_string = True
                result.append(char)
            else:
                # Potentially closing the string — check whether the following character is a valid JSON boundary
                j = i + 1
                while j < len(json_str) and json_str[j] in ' \t\r\n':
                    j += 1
                if j >= len(json_str) or json_str[j] in ':,}]':
                    # Valid boundary — a real closing quote
                    in_string = False
                    result.append(char)
                else:
                    # Invalid boundary — this is an unescaped quote inside a string
                    result.append('\\"')
            i += 1
            continue

        result.append(char)
        i += 1

    return ''.join(result)


def _remove_js_comments_outside_strings(text: str) -> str:
    """Remove JavaScript-style comments (// and /* */) only when outside JSON string values.

    The naive regex approach (re.sub(r'//[^\n]*\n', ...)) operates on raw text and
    will corrupt JSON strings that contain JavaScript code with // comments —
    e.g. optimized_code values.  This function walks character-by-character,
    tracking whether we are inside a JSON string, and only strips comments that
    appear outside strings.
    """
    result = []
    i = 0
    n = len(text)
    in_string = False

    while i < n:
        char = text[i]

        # Handle escape sequences inside strings
        if in_string and char == '\\':
            result.append(char)
            if i + 1 < n:
                i += 1
                result.append(text[i])
            i += 1
            continue

        # Toggle string state on unescaped double-quotes
        if char == '"':
            in_string = not in_string
            result.append(char)
            i += 1
            continue

        # Only strip comments when outside a string
        if not in_string and char == '/':
            # // line comment → skip to end of line, keep the newline
            if i + 1 < n and text[i + 1] == '/':
                j = i + 2
                while j < n and text[j] != '\n':
                    j += 1
                if j < n:
                    result.append('\n')   # preserve the newline
                    j += 1
                i = j
                continue

            # /* block comment */ → remove entirely
            if i + 1 < n and text[i + 1] == '*':
                j = i + 2
                while j + 1 < n:
                    if text[j] == '*' and text[j + 1] == '/':
                        j += 2
                        break
                    j += 1
                else:
                    j = n  # unclosed block comment — skip to end
                i = j
                continue

        result.append(char)
        i += 1

    return ''.join(result)


def robust_json_parse(json_str: str, context: str = "", log_func=None) -> Optional[Dict[str, Any]]:
    """
    Robust JSON parser that handles malformed JSON returned by LLMs.

    Common issues:
    1. Trailing commas
    2. Single quotes instead of double quotes
    3. Unescaped newlines
    4. Comments
    5. Incomplete JSON

    Args:
        json_str: JSON string to parse
        context: Context information (used for logging)
        log_func: Optional logging function; falls back to the module logger if not provided

    Returns:
        Parsed dict, or None if parsing fails
    """
    if not json_str:
        return None

    _log = log_func or logger.warning

    # First attempt: parse directly
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Second attempt: parse after cleaning common issues
    try:
        cleaned = json_str

        # Remove JavaScript-style comments (only outside JSON strings)
        cleaned = _remove_js_comments_outside_strings(cleaned)

        # Fix 3A: fix unescaped double quotes inside string values
        # Must run before single-quote conversion and newline escaping:
        # those operations use a simple " toggle to track in_string,
        # which requires every " to be a valid string boundary.
        # This function uses a lookahead heuristic (does not rely on in_string state).
        cleaned = _fix_unescaped_quotes_in_strings(cleaned)

        # Remove trailing commas (commas before } or ])
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

        # Replace single quotes with double quotes (but preserve single quotes inside strings)
        # This is a simplified pass; may fail on some edge cases.
        in_string = False
        result = []
        i = 0
        while i < len(cleaned):
            char = cleaned[i]
            if char == '"' and (i == 0 or cleaned[i-1] != '\\'):
                in_string = not in_string
                result.append(char)
            elif char == "'" and not in_string:
                result.append('"')
            else:
                result.append(char)
            i += 1
        cleaned = ''.join(result)

        # Escape unescaped newlines inside strings
        # In JSON string values, newlines must be \n rather than actual newlines
        in_str = False
        current_part = []
        for i, char in enumerate(cleaned):
            if char == '"' and (i == 0 or cleaned[i-1] != '\\'):
                in_str = not in_str
            if in_str and char == '\n':
                current_part.append('\\n')
            else:
                current_part.append(char)
        cleaned = ''.join(current_part)

        # Fix 2D: Fix unquoted keys: {key: "value"} → {"key": "value"}
        cleaned = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', cleaned)

        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        _log(f"[{context}] Still cannot parse JSON after cleanup: {e}")

    # Third attempt: extract key fields using regex
    try:
        result = {}

        # Try to extract fix_target
        fix_target_match = re.search(r'"fix_target"\s*:\s*\{[^}]*"fix_type"\s*:\s*"(\w+)"', json_str)
        if fix_target_match:
            result["fix_target"] = {"fix_type": fix_target_match.group(1)}

        # Try to extract error_category
        category_match = re.search(r'"error_category"\s*:\s*"(\w+)"', json_str)
        if category_match:
            result["error_category"] = category_match.group(1)

        # Try to extract specific_fix
        fix_match = re.search(r'"specific_fix"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', json_str)
        if fix_match:
            result.setdefault("fix_target", {})["specific_fix"] = fix_match.group(1).replace('\\"', '"')

        # Try to extract confidence
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', json_str)
        if conf_match:
            result["confidence"] = float(conf_match.group(1))

        # Try to extract reasoning
        reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', json_str)
        if reason_match:
            result["reasoning"] = reason_match.group(1).replace('\\"', '"')

        # Try to extract root_cause
        root_cause_match = re.search(r'"root_cause"\s*:\s*"([^"]+)"', json_str)
        if root_cause_match:
            result["root_cause"] = root_cause_match.group(1)

        # Try to extract complexity
        complexity_match = re.search(r'"complexity"\s*:\s*"(\w+)"', json_str)
        if complexity_match:
            result["complexity"] = complexity_match.group(1)

        # Try to extract affected_lines
        affected_lines_match = re.search(r'"affected_lines"\s*:\s*(\d+)', json_str)
        if affected_lines_match:
            result["affected_lines"] = int(affected_lines_match.group(1))

        # Try to extract use_incremental_edit
        incremental_match = re.search(r'"use_incremental_edit"\s*:\s*(true|false)', json_str, re.IGNORECASE)
        if incremental_match:
            result["use_incremental_edit"] = incremental_match.group(1).lower() == "true"

        if result:
            logger.info(f"[{context}] Extracted partial fields using regex: {list(result.keys())}")
            return result
    except Exception as e:
        _log(f"[{context}] Regex extraction also failed: {e}")

    return None


def extract_json_from_response(response_text: str, context: str = "") -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from an LLM response text.

    Args:
        response_text: LLM response text (may contain non-JSON content)
        context: Context information (used for logging)

    Returns:
        Extracted and parsed JSON object, or None on failure
    """
    if not response_text:
        return None

    # Fix 2E: Strip thinking tags (e.g., <think>...</think>) from reasoning models
    # Also handle unclosed <think> (e.g., truncated by max_model_len)
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    response_text = re.sub(r'<think>.*', '', response_text, flags=re.DOTALL)
    response_text = response_text.strip()
    if not response_text:
        return None

    # Try to parse the entire response directly
    result = robust_json_parse(response_text, context)
    if result:
        # If robust_json_parse only extracted a few fields via the regex fallback (e.g. just root_cause),
        # do not return immediately — later brace counting or markdown extraction may yield full JSON.
        # Only return immediately when the result contains core output fields.
        _MAIN_FIELDS = ('optimized_code', 'self_issues', 'issues', 'change_summary',
                        'reasoning', 'gradient_type', 'direction')
        if any(k in result for k in _MAIN_FIELDS):
            return result
        # Save as fallback and continue trying better strategies
        fallback_result = result
    else:
        fallback_result = None

    # === FIX2.0: improved JSON extraction, handling markdown content after the JSON ===
    # Method 1: use a brace counter to find a complete JSON object
    try:
        first_brace = response_text.find('{')
        if first_brace != -1:
            # Preprocess: fix unescaped quotes so the brace counter's in_string tracking is correct
            json_region = _fix_unescaped_quotes_in_strings(response_text[first_brace:])

            brace_count = 0
            in_string = False
            escape_next = False

            for i in range(len(json_region)):
                char = json_region[i]

                # Handle content inside strings
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue

                # Count braces only outside strings
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found the matching closing brace
                            json_str = json_region[:i+1]
                            result = robust_json_parse(json_str, context)
                            if result:
                                return result
                            # Fix 2B: Mode B fallback — strip markdown code blocks and retry
                            # LLM sometimes puts ```javascript ... ``` in suggested_fix values,
                            # which breaks JSON parsing due to unescaped braces/newlines.
                            # Replace code blocks with empty string "" to preserve JSON structure.
                            sanitized = re.sub(r'```\w*\n.*?```', '""', json_str, flags=re.DOTALL)
                            if sanitized != json_str:
                                result = robust_json_parse(sanitized, context)
                                if result:
                                    logger.info(f"[{context}] Mode B recovery: stripped code blocks to parse JSON")
                                    return result
                            break
    except Exception:
        pass  # Fall back to pattern matching

    # Try to extract a JSON block (may be wrapped in a markdown code block)
    json_patterns = [
        r'```json\s*(.*?)```',
        r'```\s*(.*?)```',
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            result = robust_json_parse(match.strip(), context)
            if result:
                return result

    # If the earlier robust_json_parse already extracted partial fields via regex, return them as a fallback
    if fallback_result:
        logger.info(
            f"[{context}] Returning regex-fallback result (partial): {list(fallback_result.keys())}"
        )
        return fallback_result

    # Blind-spot 2 fix: log diagnostics when all extraction strategies fail
    has_brace = '{' in response_text
    logger.warning(
        f"[{context}] All JSON extraction strategies failed | "
        f"response_chars={len(response_text)} | "
        f"has_brace={has_brace} | "
        f"first_100_chars={response_text[:100]!r}"
    )
    return None


class LLMInvoker:
    """Unified LLM invocation utility.

    Provides LLM call functionality with statistics tracking and robust JSON parsing.

    Usage:
        invoker = LLMInvoker(llm, stats_tracker, logger)

        # Call the LLM
        response = invoker.invoke(
            messages=[...],
            process_type="optimization",
            function_name="quick_optimize_skill",
            skill_name="craftIronSword",
        )

        # Parse JSON
        data = invoker.parse_json(response.content, context="diagnosis")

        # Extract JSON from a response
        data = invoker.extract_json(response.content, context="analysis")
    """

    def __init__(self, llm, stats_tracker=None, instance_logger=None):
        """
        Initialize the LLMInvoker.

        Args:
            llm: LangChain LLM instance
            stats_tracker: Optional statistics tracker
            instance_logger: Optional logger instance
        """
        self.llm = llm
        self.stats_tracker = stats_tracker
        self.logger = instance_logger or logger

    def invoke(
        self,
        messages: List["BaseMessage"],
        process_type: str,
        function_name: str,
        task: str = None,
        skill_name: str = None,
        metadata: Dict[str, Any] = None,
    ):
        """
        Invoke the LLM and record statistics.

        Args:
            messages: List of LLM messages
            process_type: Process type
            function_name: Function name
            task: Associated task
            skill_name: Associated skill name
            metadata: Additional metadata

        Returns:
            LLM response object

        Raises:
            Exception: re-raises the original exception when the LLM call fails
        """
        try:
            response = self.llm.invoke(messages)

            # Record statistics (if stats_tracker is provided)
            if self.stats_tracker:
                token_usage = self.stats_tracker.extract_token_usage(response)
                self.stats_tracker.record_llm_call(
                    process_type=process_type,
                    function_name=function_name,
                    task=task,
                    skill_name=skill_name,
                    input_tokens=token_usage.get("input_tokens"),
                    output_tokens=token_usage.get("output_tokens"),
                    total_tokens=token_usage.get("total_tokens"),
                    model_name=self.llm.model_name,
                    success=True,
                    metadata=metadata,
                )

            return response
        except Exception as e:
            # Record the failed call
            if self.stats_tracker:
                self.stats_tracker.record_llm_call(
                    process_type=process_type,
                    function_name=function_name,
                    task=task,
                    skill_name=skill_name,
                    model_name=self.llm.model_name,
                    success=False,
                    error_message=str(e),
                    metadata=metadata,
                )
            raise

    def parse_json(self, json_str: str, context: str = "") -> Optional[Dict[str, Any]]:
        """
        Robust JSON parsing.

        Args:
            json_str: JSON string to parse
            context: Context information (used for logging)

        Returns:
            Parsed dict, or None on failure
        """
        return robust_json_parse(json_str, context, log_func=self.logger.warning)

    def extract_json(self, response_text: str, context: str = "") -> Optional[Dict[str, Any]]:
        """
        Extract a JSON object from an LLM response text.

        Args:
            response_text: LLM response text
            context: Context information (used for logging)

        Returns:
            Extracted and parsed JSON object, or None on failure
        """
        return extract_json_from_response(response_text, context)
