"""The wire types of the tool-call refusals a vLLM tree serves.

A tool call the model wrote that cannot be served is refused with a member of
ToolCallRefusedError, and the class requires every member to declare, as a
class keyword, the wire_type that names it on the wire -- the error type a
whole response and a stream's error event carry, by which a caller tells the
model's refused call from every other error. This reads every member's
wire_type from the source of the tree it is given and prints them, sorted and
space-separated: the refusals the image built from that tree serves. The lock
pins exactly this (SERVED_TOOL_CALL_REFUSAL_TYPES), and the paired client
holds its table of the refusals it redraws to the pin, so a member added,
renamed or removed on one side alone is refused by the other's gate.

A class that declares a wire_type is a member only through ToolCallRefusedError
or another member, by the name it is defined under; a wire_type that is not one
name, a member whose base is neither, and two members one type names are
refused, since each would be a refusal no caller can tell apart.
"""

import ast
import sys
from pathlib import Path

BASE = "ToolCallRefusedError"


def refuse(message: str) -> None:
    sys.exit(f"served_refusal_types: {message}")


def members(root: Path) -> dict[str, tuple[str, str]]:
    """Each member's class name -> (wire type, where it is defined)."""
    declared: dict[str, tuple[list[str], str, str]] = {}
    for path in sorted((root / "vllm").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "wire_type" not in source:
            continue
        where = str(path.relative_to(root))
        for node in ast.walk(ast.parse(source, filename=where)):
            if not isinstance(node, ast.ClassDef):
                continue
            keywords = [k for k in node.keywords if k.arg == "wire_type"]
            if not keywords:
                continue
            value = keywords[0].value
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)
                    and value.value.isidentifier()):
                refuse(f"{where}:{node.lineno}: {node.name} declares a wire_type "
                       "that is not one name")
            bases = [base.id for base in node.bases if isinstance(base, ast.Name)]
            if node.name in declared:
                refuse(f"{where}:{node.lineno}: {node.name} is defined twice")
            declared[node.name] = (bases, value.value, f"{where}:{node.lineno}")
    found: dict[str, tuple[str, str]] = {}
    while True:
        joined = {
            name: (wire, where)
            for name, (bases, wire, where) in declared.items()
            if name not in found and any(b == BASE or b in found for b in bases)
        }
        if not joined:
            break
        found.update(joined)
    for name, (bases, _, where) in declared.items():
        if name not in found:
            refuse(f"{where}: {name} declares a wire_type but is not a member of "
                   f"{BASE}: its bases are {bases}")
    return found


def main() -> None:
    if len(sys.argv) != 2:
        refuse("usage: served_refusal_types.py TREE")
    found = members(Path(sys.argv[1]))
    if not found:
        refuse(f"the tree declares no member of {BASE}")
    types: dict[str, str] = {}
    for name, (wire, where) in sorted(found.items()):
        if wire in types:
            refuse(f"{where}: {name} and {types[wire]} are both served as {wire}")
        types[wire] = name
    print(" ".join(sorted(types)))


if __name__ == "__main__":
    main()
