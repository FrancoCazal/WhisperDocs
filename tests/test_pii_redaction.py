"""No email, phone or ID number in the query log (audit M-5).

`redact` (app/services/pii.py) finds each kind and leaves alone the numbers a
policy answer is made of: dates, times, ports, amounts, versions, IPs and
ranges. A number with a check digit is named only when the digit is right.
`QueryLog` redacts its question and answer on every write.
"""

from __future__ import annotations

import pytest

from app.models import QueryLog
from app.services.pii import cbu_valid, cuit_valid, luhn_valid, redact

# Check digits computed by hand: CUIT 20-12345678 → 6; the CBU's blocks
# 2850590 → 9 and 9400901234567 → 4.
CUIT = "20-12345678-6"
CBU = "2850590994009012345674"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Escribime a juan.perez+rrhh@empresa.com.ar", "Escribime a [email]"),
        ("o al +54 9 11 5555-5555.", "o al [phone]."),
        ("Call +1 (415) 555-0100 or 1155555555.", "Call [phone] or [phone]."),
        ("Llamá al (011) 4555-5555.", "Llamá al [phone]."),
        ("Mi tel: 4555-5555, celular 15 5555 5555", "Mi tel: [phone], celular [phone]"),
        ("Card 4111 1111 1111 1111.", "Card [card]."),
        ("Card 4111-1111-1111-1111.", "Card [card]."),
        (f"CBU {CBU}", "CBU [cbu]"),
        (f"CUIT {CUIT}", "CUIT [cuit]"),
        ("CUIL 20123456786", "CUIL [cuit]"),
        ("DNI 12.345.678 y DNI n° 7654321", "DNI [dni] y DNI n° [dni]"),
    ],
)
def test_each_kind_is_replaced(text, expected):
    assert redact(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Invoices are kept for 5 years and tickets for 2 years.",
        "The VPN uses UDP port 51820; MFA closes it after 30 seconds.",
        "Due on 2024-01-15 10:30, or on 15-01-2024.",
        "On 2024-01-15 12 people came.",
        "A budget of 1234567890.50 dollars.",
        "From 2020 - 2024, and 2020-2024; reviewed in 2019 - 2021 - 2023.",
        "Up to $1.500.000, or 1,000,000,000 pesos, at 3.14159265358 percent.",
        "Version 1.2.3 runs on 192.168.100.200.",
        "Order 12345678 shipped.",
        # Sixteen digits that fail Luhn: a tracking number, not a card.
        "Tracking 4111-1111-1111-1112.",
    ],
)
def test_the_numbers_an_answer_is_made_of_stay(text):
    assert redact(text) == text


def test_a_check_digit_decides_what_a_number_is():
    assert luhn_valid("4111111111111111") and not luhn_valid("4111111111111112")
    assert cuit_valid("20123456786") and not cuit_valid("20123456780")
    # The check digit is right (99-12345678 → 1), but 99 is no CUIT prefix.
    assert not cuit_valid("99123456781")
    assert cbu_valid(CBU) and not cbu_valid(CBU[:-1] + "5")
    assert not cbu_valid("28505901" + "0" * 14)


def test_an_eleven_digit_number_with_a_wrong_check_digit_is_still_redacted():
    """Not a CUIT, but ten or more digits read as a phone."""
    assert redact("CUIL 20123456780") == "CUIL [phone]"


def test_the_query_log_redacts_on_every_write():
    log = QueryLog(question="soy ana@example.com", answer="call +54 9 11 5555-5555")

    assert (log.question, log.answer) == ("soy [email]", "call [phone]")
    log.answer = f"CUIT {CUIT}"
    assert log.answer == "CUIT [cuit]"
