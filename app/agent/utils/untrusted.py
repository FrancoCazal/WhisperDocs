"""Untrusted text in a prompt: documents, a summary written from them, a
transcript of what users and the model said (audit A10).

Each goes inside an element named for it (`<documents>`, `<transcript>`),
and the system prompts say the element holds data, not instructions. The text
could otherwise end the element itself: a document holding `</documents>`
and then a forged question would read as the prompt's own. `fill` makes every
tag of that element's name inside the text inert, its `<` turned into
`&lt;`, so the only closing tag left is the prompt's.
"""

from __future__ import annotations

import re

# A placeholder alone in an element of its template: "<documents>\n{context}\n</documents>".
_ELEMENT = re.compile(r"<(\w+)>\n\{(\w+)\}\n</\1>")


def escaped(tag: str, text: str) -> str:
    """`text` with each opening or closing `tag` tag made inert, in any case
    and with spaces inside the brackets (`< / Documents >`)."""
    return re.sub(rf"<(?=\s*/?\s*{re.escape(tag)}\b)", "&lt;", text, flags=re.IGNORECASE)


def elements(template: str) -> dict[str, str]:
    """Each placeholder that sits alone in an element of `template`, with
    the element's name: {"context": "documents"}."""
    return {field: tag for tag, field in _ELEMENT.findall(template)}


def fill(template: str, **values: str) -> str:
    """`template.format(**values)`, each value that sits alone in an element
    of the template escaped against that element's tag."""
    held = elements(template)
    return template.format(
        **{
            name: escaped(held[name], value) if name in held else value
            for name, value in values.items()
        }
    )
