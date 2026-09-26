"""Optional multi-turn contract from challenge-brief §7.4.

    respond(state: ConversationState, merchant_message: str) -> dict

`state` carries short-term memory (turns, intents, auto-reply count, actions, exit state) and
references to the long-term contexts. Returns {"action": "send"|"wait"|"end", ...}.
"""
from __future__ import annotations

from typing import Optional

from vera.conversation import ConversationState, ReplyEngine

_ENGINE = ReplyEngine()


def new_state(conversation_id: str, category: dict, merchant: dict, trigger: Optional[dict] = None,
              customer: Optional[dict] = None, first_message: Optional[str] = None) -> ConversationState:
    st = ConversationState(conversation_id=conversation_id, merchant_id=merchant.get("merchant_id"),
                           customer_id=(customer or {}).get("customer_id"), trigger_id=(trigger or {}).get("id"),
                           category=category, merchant=merchant, trigger=trigger or {}, customer=customer)
    if first_message:
        st.record_bot(first_message, "open_ended")
    return st


def respond(state: ConversationState, merchant_message: str) -> dict:
    return _ENGINE.respond(state, merchant_message, "customer" if state.customer else "merchant")


__all__ = ["ConversationState", "new_state", "respond"]
