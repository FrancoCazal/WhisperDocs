import uuid

# Fixed namespace — NEVER change, invalidates all existing WhatsApp thread IDs.
WHATSAPP_NAMESPACE = uuid.UUID("a1b2c3d4-5678-4abc-def0-123456789abc")


def normalize_phone(raw: str) -> str:
    """'whatsapp:+595981234567' → '595981234567'"""
    return raw.replace("whatsapp:", "").replace("+", "").strip()


def whatsapp_thread_id(phone_raw: str) -> str:
    """Deterministic thread_id for a WhatsApp phone. Same phone → same UUID always.

    UUID5 keeps PII out of thread_ids (logs, URLs, checkpointer tables).
    The plain phone is stored in Conversation.user_number only — see phase-4-design §2.7.
    """
    return str(uuid.uuid5(WHATSAPP_NAMESPACE, normalize_phone(phone_raw)))
