"""Personal data out of the query log (audit M-5).

`query_logs` keeps each question and answer for the history, the metrics and
debugging. None of those needs the email, phone or ID number a user typed, or
one an answer quoted from a document. `redact` puts a placeholder in its
place: `[email]`, `[phone]`, `[card]`, `[cbu]`, `[cuit]` or `[dni]`.
`QueryLog` redacts on assignment, so every write goes through it.

A card (by Luhn), a CBU and a CUIT or CUIL are told apart by their check
digits. A phone number has none: a number counts as one when it starts with
`+` and has 8 to 15 digits, when it has 10 to 15, or when it has 6 or more
after a word like "tel" or "celular". So now and then the log loses an
order number of that length, rather than keep a phone. A DNI counts only
after the word "DNI": eight bare digits are as often an amount or a code.
Dates, times, decimals, IP addresses and year ranges stay.

What keeps the words as written: the thread in the checkpointer, which the
agent reads back on the next turn, and the conversation's title, the user's
own first question.
"""

from __future__ import annotations

import re

EMAIL = "[email]"
PHONE = "[phone]"
CARD = "[card]"
CBU = "[cbu]"
CUIT = "[cuit]"
DNI = "[dni]"

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# "DNI 12.345.678", "DNI n° 12345678", "DNI: 7.654.321".
_DNI = re.compile(r"(?i)(\bDNI\b\s*(?:n[°º.o]*\s*)?:?\s*)\d{1,2}\.?\d{3}\.?\d{3}\b")
# A run of digits with a space, a hyphen or parentheses between groups, and
# an optional `+`: "+54 9 11 5555-5555", "(011) 4555-5555". Not inside a
# word, a decimal or a dotted number (an IP, an amount), not followed by a
# time's `:` or a decimal's point, and not across " - ", which a range has.
_NUMBER = re.compile(r"(?<![\w.,+])\+?\(?\d(?:\)?[ \-]?\(?\d)*(?![\w])(?![:.,]\d)")
_PHONE_WORD = re.compile(
    r"(?i)\b(?:tel|tel[eé]fono|cel|celular|m[oó]vil|phone|mobile|cell|whatsapp|wsp|fax)\b"
    r"\.?\s*(?:n[°º.o]*\s*)?:?\s*$"
)
_DATE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}-\d{1,2}-\d{4}")

_CUIT_PREFIXES = {"20", "23", "24", "27", "30", "33", "34"}


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def cuit_valid(digits: str) -> bool:
    """A CUIT or CUIL: 11 digits, a known prefix, and the mod 11 check digit."""
    if len(digits) != 11 or digits[:2] not in _CUIT_PREFIXES:
        return False
    total = sum(int(d) * w for d, w in zip(digits, (5, 4, 3, 2, 7, 6, 5, 4, 3, 2), strict=False))
    check = 11 - total % 11
    check = 0 if check == 11 else check
    return check != 10 and check == int(digits[10])


def _block_valid(block: str, weights: tuple[int, ...]) -> bool:
    total = sum(int(d) * w for d, w in zip(block, weights, strict=False))
    return (10 - total % 10) % 10 == int(block[-1])


def cbu_valid(digits: str) -> bool:
    """An Argentine CBU: 22 digits in two blocks, each with its check digit."""
    return (
        len(digits) == 22
        and _block_valid(digits[:8], (7, 1, 3, 9, 7, 1, 3))
        and _block_valid(digits[8:], (3, 9, 7, 1, 3, 9, 7, 1, 3, 9, 7, 1, 3))
    )


def _number(match: re.Match[str]) -> str:
    text = match.group(0)
    digits = re.sub(r"\D", "", text)
    n = len(digits)
    if 13 <= n <= 19 and luhn_valid(digits):
        return CARD
    if cbu_valid(digits):
        return CBU
    if cuit_valid(digits):
        return CUIT
    if _DATE.search(text):
        return text
    before = match.string[max(0, match.start() - 30) : match.start()]
    phone = (
        (text.startswith("+") and n >= 8)
        or n >= 10
        or (n >= 6 and _PHONE_WORD.search(before) is not None)
    )
    if phone and n <= 15:
        return PHONE
    return text


def redact(text: str) -> str:
    """`text` with each email, phone, card, CBU, CUIT/CUIL and DNI replaced
    by its placeholder."""
    text = _EMAIL.sub(EMAIL, text)
    text = _DNI.sub(lambda m: m.group(1) + DNI, text)
    return _NUMBER.sub(_number, text)
