"""
Parameter Parser - Babel AST parameter-parsing utility

Pure functions extracted from graph_manager_impl.py for parsing JavaScript function parameters.
Supports extracting parameter types, default values, etc. from Babel AST nodes.

v4.0 modular refactor
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def extract_destructured_params(object_pattern) -> Dict[str, Dict[str, Any]]:
    """
    Extract destructured parameters from a Babel AST ObjectPattern node.

    Supports forms like: { toolName = "wooden_axe", count = 1, plankName = "oak_planks" } = {}

    Args:
        object_pattern: Babel AST ObjectPattern node

    Returns:
        parameter metadata dict
    """
    parameters = {}

    try:
        # Get the properties of ObjectPattern
        properties = getattr(object_pattern, 'properties', None)
        if not properties:
            return parameters

        for prop in properties:
            try:
                prop_type = getattr(prop, 'type', None)

                # Handle ObjectProperty or Property
                if prop_type in ("ObjectProperty", "Property"):
                    key = getattr(prop, 'key', None)
                    value = getattr(prop, 'value', None)

                    if not key:
                        continue

                    # Get the property name
                    prop_name = None
                    if getattr(key, 'type', None) == "Identifier":
                        prop_name = key.name
                    elif hasattr(key, 'name'):
                        prop_name = key.name

                    if not prop_name:
                        continue

                    # Get the default value (if any)
                    param_type = "unknown"
                    param_default = None

                    # Check whether value is an AssignmentPattern (has a default value)
                    if value and getattr(value, 'type', None) == "AssignmentPattern":
                        right = getattr(value, 'right', None)
                        if right:
                            param_type = infer_type_from_node(right)
                            param_default = extract_value_from_node(right)
                    # Check whether value is an Identifier (no default value)
                    elif value and getattr(value, 'type', None) == "Identifier":
                        # No default value
                        param_type = "unknown"
                        param_default = None

                    param_info = {
                        "type": param_type,
                        "default": param_default,
                        "description": f"Parameter '{prop_name}' from destructured object"
                    }

                    parameters[prop_name] = param_info

                # Handle shorthand properties (e.g., { toolName } without default)
                elif prop_type == "Identifier":
                    prop_name = getattr(prop, 'name', None)
                    if prop_name:
                        parameters[prop_name] = {
                            "type": "unknown",
                            "default": None,
                            "description": f"Parameter '{prop_name}' from destructured object"
                        }

            except Exception as e:
                logger.warning(f"[Parameter Extraction] Failed to extract destructured property: {e}")
                continue

    except Exception as e:
        logger.warning(f"[Parameter Extraction] Failed to extract destructured params: {e}")

    return parameters


def infer_type_from_node(node) -> str:
    """Infer the type from an AST node."""
    if not node or not hasattr(node, 'type'):
        return "unknown"

    try:
        node_type = node.type
        if node_type == "NumericLiteral":
            return "number"
        elif node_type == "StringLiteral":
            return "string"
        elif node_type == "BooleanLiteral":
            return "boolean"
        elif node_type == "ArrayExpression":
            return "array"
        elif node_type == "ObjectExpression":
            return "object"
        elif node_type == "NullLiteral":
            return "null"
    except Exception:
        pass

    return "unknown"


def extract_value_from_node(node) -> Any:
    """Extract a value from an AST node."""
    if not node or not hasattr(node, 'type'):
        return None

    try:
        node_type = node.type

        if node_type in ("NumericLiteral", "StringLiteral", "BooleanLiteral"):
            return getattr(node, 'value', None)

        elif node_type == "NullLiteral":
            return None

        elif node_type == "ArrayExpression":
            elements = getattr(node, 'elements', [])
            if elements:
                return [extract_value_from_node(el) for el in elements if el]
            return []

        elif node_type == "ObjectExpression":
            # Return an empty object to indicate the default value is an empty object
            return {}

    except Exception:
        pass

    return None


def infer_param_type_from_babel(param_node) -> str:
    """
    Infer the parameter type from a Babel AST node.

    Args:
        param_node: Babel AST parameter node

    Returns:
        Parameter type string
    """
    # Check whether there is a default value
    if not hasattr(param_node, 'right') or param_node.right is None:
        return "unknown"

    default_node = param_node.right

    # Defensive check: ensure default_node has a type attribute
    if not hasattr(default_node, 'type'):
        return "unknown"

    try:
        # Check the node type
        node_type = default_node.type
        if node_type == "NumericLiteral":
            return "number"
        elif node_type == "StringLiteral":
            return "string"
        elif node_type == "BooleanLiteral":
            return "boolean"
        elif node_type == "ArrayExpression":
            return "array"
        elif node_type == "ObjectExpression":
            return "object"
        elif node_type == "NullLiteral":
            # A null default cannot infer a type from syntax; return nullable to trigger subsequent semantic analysis
            return "nullable"
        else:
            # Try inferring from the value
            if hasattr(default_node, 'value'):
                value = default_node.value
                if isinstance(value, (int, float)):
                    return "number"
                elif isinstance(value, str):
                    return "string"
                elif isinstance(value, bool):
                    return "boolean"
                elif isinstance(value, list):
                    return "array"
                elif isinstance(value, dict):
                    return "object"
    except Exception as e:
        logger.warning(f"[Parameter Extraction] Failed to infer param type: {e}")

    return "unknown"


def extract_default_value_from_babel(param_node) -> Any:
    """
    Extract the default value from a Babel AST node.

    Args:
        param_node: Babel AST parameter node

    Returns:
        Default value (Python object)
    """
    if not hasattr(param_node, 'right') or param_node.right is None:
        return None

    default_node = param_node.right

    # Defensive check: ensure default_node has a type attribute
    if not hasattr(default_node, 'type'):
        return None

    try:
        node_type = default_node.type
        if node_type == "NumericLiteral":
            return default_node.value if hasattr(default_node, 'value') else None
        elif node_type == "StringLiteral":
            return default_node.value if hasattr(default_node, 'value') else None
        elif node_type == "BooleanLiteral":
            return default_node.value if hasattr(default_node, 'value') else None
        elif node_type == "ArrayExpression":
            # Extract array elements
            if not hasattr(default_node, 'elements'):
                return None
            elements = []
            for elem in default_node.elements:
                if elem is None:
                    continue
                if not hasattr(elem, 'type'):
                    continue
                try:
                    if elem.type == "StringLiteral" and hasattr(elem, 'value'):
                        elements.append(elem.value)
                    elif elem.type == "NumericLiteral" and hasattr(elem, 'value'):
                        elements.append(elem.value)
                    elif elem.type == "BooleanLiteral" and hasattr(elem, 'value'):
                        elements.append(elem.value)
                except Exception as e:
                    logger.warning(f"[Parameter Extraction] Failed to extract array element: {e}")
                    continue
            return elements
        elif node_type == "ObjectExpression":
            # Extract object properties
            if not hasattr(default_node, 'properties'):
                return None
            obj = {}
            for prop in default_node.properties:
                if prop is None:
                    continue
                try:
                    if hasattr(prop, 'key') and hasattr(prop.key, 'name') and prop.key.name:
                        if hasattr(prop, 'value'):
                            obj[prop.key.name] = extract_value_from_babel_node(prop.value)
                except Exception as e:
                    logger.warning(f"[Parameter Extraction] Failed to extract object property: {e}")
                    continue
            return obj
        else:
            # Try fetching the value directly
            if hasattr(default_node, 'value'):
                return default_node.value
    except Exception as e:
        logger.warning(f"[Parameter Extraction] Failed to extract default value: {e}")
        import traceback
        logger.debug(f"[Parameter Extraction] Traceback: {traceback.format_exc()}")

    return None


def extract_value_from_babel_node(node) -> Any:
    """Recursively extract the value of a Babel node."""
    if node is None or not hasattr(node, 'type'):
        return None

    try:
        node_type = node.type
        if node_type == "NumericLiteral":
            return node.value if hasattr(node, 'value') else None
        elif node_type == "StringLiteral":
            return node.value if hasattr(node, 'value') else None
        elif node_type == "BooleanLiteral":
            return node.value if hasattr(node, 'value') else None
        elif node_type == "ArrayExpression":
            if not hasattr(node, 'elements'):
                return None
            return [extract_value_from_babel_node(elem) for elem in node.elements if elem is not None]
        elif node_type == "ObjectExpression":
            if not hasattr(node, 'properties'):
                return None
            result = {}
            for prop in node.properties:
                if prop is None:
                    continue
                try:
                    if hasattr(prop, 'key') and hasattr(prop.key, 'name') and prop.key.name:
                        if hasattr(prop, 'value'):
                            result[prop.key.name] = extract_value_from_babel_node(prop.value)
                except:
                    continue
            return result
    except Exception as e:
        logger.warning(f"[Parameter Extraction] Failed to extract value from node: {e}")

    return None


def generate_param_description(param_name: str, param_node=None) -> str:
    """
    Generate a parameter description (inferred from the parameter name).

    Args:
        param_name: Parameter name
        param_node: Babel AST node (optional)

    Returns:
        Parameter description string
    """
    # Simple inference based on the parameter name
    param_lower = param_name.lower()

    if "count" in param_lower or "quantity" in param_lower or "num" in param_lower:
        return "Number of items"
    elif "type" in param_lower or "kind" in param_lower:
        return "Type of item"
    elif "name" in param_lower:
        return "Name of the item"
    elif "log" in param_lower and "type" in param_lower:
        return "List of allowed log types"
    elif "log" in param_lower and "types" in param_lower:
        return "List of allowed log types"
    elif "allowed" in param_lower:
        return "List of allowed values"
    elif "distance" in param_lower:
        return "Distance in blocks"
    elif "time" in param_lower or "timeout" in param_lower:
        return "Time duration in seconds"
    elif "direction" in param_lower:
        return "Direction vector"
    else:
        return f"Parameter {param_name}"
