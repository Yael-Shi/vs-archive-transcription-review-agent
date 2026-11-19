"""Transcript checks and reply extraction for the Grok renderer.

The JavaScript below follows the selectors used in the pipe pilot. Activity
rows and ``data-grok-state`` marks are not treated as message text. Reply
text is copied from ``.sand-message-prose`` inside the tagged user turn's
transcript rows. It is not chosen by what the words say. Grok 1.0.1 puts
each entry in its own ``.sand-transcript-row``, so state is not read from
the user message's parent. A trailing ``.sand-virtual-transcript__inset``
ends that rendered list. The same inset with a transcript row after it, or
any other non-row sibling, leaves the turn ambiguous.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Any

_AGENT_ID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,63}$")
_ACTIVITY = frozenset(
    {
        "thinking",
        "searching",
        "reading",
        "orbit",
        "writing",
        "coding",
        "progress",
        "running",
        "computing",
        "controlling",
        "working",
        "sending",
        "waiting",
        "streaming",
    }
)
_RUNNING = _ACTIVITY
_KNOWN_STATES = _RUNNING | {"failed", "blocked", "finished", "idle"}
_MAX_MESSAGE = 2000
FILE_INPUT_SELECTOR = "input.sand-prompt-file-input"
_CHROME_CLASSES = frozenset(
    {
        "sand-activity-mark",
        "sand-activity-line",
        "sand-message-typing",
        "sand-message-attachments",
    }
)
_VOID_TAGS = frozenset({"br", "hr"})
_INLINE_TAGS = frozenset(
    {
        "a",
        "abbr",
        "b",
        "cite",
        "code",
        "del",
        "em",
        "i",
        "kbd",
        "mark",
        "s",
        "samp",
        "small",
        "span",
        "strong",
        "sub",
        "sup",
        "u",
        "var",
    }
)
# Paragraph boundaries follow the blocks in Grok 0.79.4's prose copy walker.
_PARAGRAPH_TAGS = frozenset(
    {"p", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "li", "hr"}
)
_WRAPPER_TAGS = frozenset({"ul", "ol", "section", "article"})
MAX_REPLY = 200_000
PREVIEW_CHARS = 160


def validate_request(
    *,
    agent_id: str,
    bot_name: str,
    message: str,
    request_id: str,
) -> str | None:
    if not _AGENT_ID.fullmatch(agent_id):
        return "bad_agent_id"
    name = " ".join(bot_name.split())
    if not name or len(name) > 80:
        return "bad_bot_name"
    if not _REQUEST_ID.fullmatch(request_id):
        return "bad_request_id"
    if not message or len(message) > _MAX_MESSAGE or request_id not in message:
        return "bad_message"
    return None


def as_list(value: Any) -> list[str]:
    """Normalize a CDP value that PowerShell may collapse to one string."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def completion_phase(states: list[str] | tuple[str, ...]) -> str:
    """Reduce one turn's ``data-grok-state`` values to one phase.

    The caller has to pass states attributed to that turn. ``failed`` and
    ``blocked`` win. A running mark wins over ``finished``, so a stream that
    is still active is not completion. ``idle`` alone is not completion.
    """

    values = [str(item) for item in states]
    if any(item not in _KNOWN_STATES for item in values):
        return "unknown"
    known = set(values)
    if "failed" in known:
        return "failed"
    if "blocked" in known:
        return "blocked"
    if known & _RUNNING:
        return "running"
    if "finished" in known:
        return "finished"
    return "idle"


def _flag(poll: dict[str, Any], name: str) -> bool:
    return poll.get(name) is True


def judge_turn(poll: dict[str, Any], agent_id: str) -> tuple[str, str]:
    """Return ``(phase, refusal)`` for one poll of the current turn.

    States, send flags, and the streaming ``aria-label`` have to come from
    the rows of this user turn. A sidebar ``finished`` mark is not a phase.
    ``settled`` means the turn's own assistant prose is present, the
    renderer is not streaming it, and no in-range activity or send flag is
    still active. Identical text is not itself a phase.
    """

    attribution = str(poll.get("attribution") or "")
    if attribution == "virtualized":
        return "ambiguous", "virtualized_turn"
    if attribution == "ambiguous":
        return "ambiguous", "ambiguous_completion"
    if attribution != "turn":
        return "idle", ""
    active = str(poll.get("activeAgent") or "")
    if active != agent_id:
        return "ambiguous", "completion_wrong_bot"
    phase = completion_phase(as_list(poll.get("states")))
    if phase == "unknown":
        return "ambiguous", "unknown_completion_state"
    if _flag(poll, "failed") or phase == "failed":
        return "failed", ""
    if phase == "blocked":
        return "blocked", ""
    streaming = _flag(poll, "streaming")
    sending = _flag(poll, "pending") or _flag(poll, "busy")
    if phase == "finished" and (streaming or sending):
        return "running", ""
    if streaming or sending or phase == "running":
        return "running", ""
    if phase == "finished":
        return "finished", ""
    if _turn_settled(poll):
        return "settled", ""
    return "idle", ""


def _turn_settled(poll: dict[str, Any]) -> bool:
    """True when this turn's renderer signals show generation has ended.

    ``streaming`` is an ``aria-label`` on the assistant message. The
    installed renderer omits that label only for finished non-empty prose.
    ``pending`` and ``busy`` are the row send flags (``data-pending`` /
    ``aria-busy``), not generation. An in-range activity mark, including
    ``idle`` or a ``data-activity`` line such as ``reading``, means the turn
    is not settled. Missing flags are not treated as false.
    """

    if poll.get("replyStructure") != "prose":
        return False
    reply = poll.get("reply")
    if not isinstance(reply, str) or reply == "":
        return False
    for name in ("streaming", "pending", "failed", "busy"):
        if poll.get(name) is not False:
            return False
    states = as_list(poll.get("states"))
    if states:
        return False
    return True


class _DomNode:
    def __init__(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tag = tag
        self.attrs = {key: value or "" for key, value in attrs}
        self.classes = frozenset(self.attrs.get("class", "").split())
        self.children: list[_DomNode | str] = []


class _DomBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _DomNode("root", [])
        self._stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _DomNode(tag, attrs)
        self._stack[-1].children.append(node)
        if tag not in _VOID_TAGS:
            self._stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        if len(self._stack) > 1 and self._stack[-1].tag == tag:
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        self._stack[-1].children.append(data)


def _is_chrome(node: _DomNode) -> bool:
    if node.classes & _CHROME_CLASSES:
        return True
    if any(
        name in node.attrs
        for name in ("data-activity", "data-grok-state", "data-typing")
    ):
        return True
    return node.tag == "time" or "hidden" in node.attrs


def _is_content_assistant(node: _DomNode) -> bool:
    return (
        "sand-message" in node.classes
        and node.attrs.get("data-role") == "assistant"
        and not _is_chrome(node)
    )


def _walk(node: _DomNode) -> list[_DomNode]:
    found = [node]
    for child in node.children:
        if isinstance(child, _DomNode):
            found.extend(_walk(child))
    return found


def _render_inline(node: _DomNode) -> str | None:
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, str):
            parts.append(child)
            continue
        if _is_chrome(child):
            continue
        if child.tag == "br":
            parts.append("\n")
            continue
        if child.tag in _INLINE_TAGS:
            inner = _render_inline(child)
            if inner is None:
                return None
            parts.append(inner)
            continue
        return None
    return "".join(parts)


def _is_structural(node: _DomNode) -> bool:
    return (
        node.tag in _PARAGRAPH_TAGS
        or node.tag == "div"
        or node.tag in _WRAPPER_TAGS
        or node.tag == "table"
    )


def _blocks_of(node: _DomNode) -> list[str] | None:
    structural = [
        child
        for child in node.children
        if isinstance(child, _DomNode)
        and not _is_chrome(child)
        and _is_structural(child)
    ]
    has_block = bool(structural)
    blocks: list[str] = []
    inline: list[str] = []

    def flush() -> bool:
        if not inline:
            return True
        text = "".join(inline)
        inline.clear()
        if has_block and text.strip() == "":
            if "\n" in text:
                return False
            return True
        if has_block:
            return False
        blocks.append(text)
        return True

    for child in node.children:
        if isinstance(child, str):
            if has_block and child.strip() == "":
                continue
            inline.append(child)
            continue
        if _is_chrome(child):
            continue
        if child.tag == "br":
            inline.append("\n")
            continue
        if child.tag in _INLINE_TAGS:
            inner = _render_inline(child)
            if inner is None:
                return None
            inline.append(inner)
            continue
        if child.tag == "table":
            return None
        if child.tag in _WRAPPER_TAGS or child.tag == "div":
            if not flush():
                return None
            inner_blocks = _blocks_of(child)
            if inner_blocks is None:
                return None
            transparent = child.tag in _WRAPPER_TAGS or any(
                isinstance(item, _DomNode)
                and not _is_chrome(item)
                and _is_structural(item)
                for item in child.children
            )
            if transparent:
                blocks.extend(inner_blocks)
            elif len(inner_blocks) == 1:
                blocks.append(inner_blocks[0])
            elif inner_blocks:
                return None
            continue
        if child.tag in _PARAGRAPH_TAGS:
            if not flush():
                return None
            if child.tag == "hr":
                blocks.append("")
                continue
            text = _render_inline(child)
            if text is None:
                return None
            blocks.append(text)
            continue
        return None
    if not flush():
        return None
    return blocks


def _render_prose(node: _DomNode) -> str | None:
    blocks = _blocks_of(node)
    if blocks is None:
        return None
    return "\n\n".join(blocks)


def structural_message_text(html: str) -> tuple[str, str]:
    """Copy one assistant prose node, including paragraph boundaries.

    Grok 0.79.4 renders assistant text in ``.sand-message-prose`` and copies
    ``p`` and the other blocks in its prose walker as separate paragraphs.
    Those blocks are joined with a blank line. ``<br>`` stays a single line
    break. Text inside inline tags is copied unchanged. A ``div`` that only
    wraps those blocks is transparent; a ``div`` that holds inline text is
    one paragraph. Markup that mixes those forms, or a table, is
    ``inconclusive`` and no text is returned.
    """

    builder = _DomBuilder()
    builder.feed(html)
    builder.close()
    messages = [node for node in _walk(builder.root) if _is_content_assistant(node)]
    if not messages:
        return "", "absent"
    if len(messages) != 1:
        return "", "inconclusive"
    prose = [
        node
        for node in _walk(messages[0])
        if "sand-message-prose" in node.classes and not _is_chrome(node)
    ]
    if len(prose) != 1:
        return "", "inconclusive"
    rendered = _render_prose(prose[0])
    if rendered is None:
        return "", "inconclusive"
    return rendered, "prose"


def _explicit_false(value: dict[str, Any], name: str) -> bool:
    return value.get(name) is False


def classify_preflight(value: Any) -> str | None:
    """Return a refusal reason, or None when a greeting-only chat may be sent.

    A new bot may show more than one automatic assistant message. The chat
    is usable only when every message is an assistant greeting, no user turn
    exists, and no send or streaming flag is set. One observation is not
    enough; the caller compares a second observation before sending.
    """

    if not isinstance(value, dict):
        return "preflight_invalid"
    if value.get("active") is not True:
        return "not_active"
    if value.get("nameMatch") is not True:
        return "wrong_bot"
    if value.get("composerEmpty") is not True:
        return "unexpected_draft"
    roles = as_list(value.get("roles"))
    if (
        int(value.get("userCount") or 0)
        or int(value.get("userRows") or 0)
        or "user" in roles
    ):
        return "prior_user_turns"
    if value.get("tagged") is True:
        return "prior_test_activity"
    if value.get("state") != "idle":
        return "not_idle"
    if value.get("transcript") == "virtualized":
        return "virtualized_transcript"
    if value.get("transcript") != "ready":
        return "ambiguous_transcript"
    if any(role != "assistant" for role in roles):
        return "unknown_role"
    assistants = int(value.get("assistantCount") or 0)
    messages = int(value.get("msgCount") or 0)
    if assistants < 1 or messages != assistants or len(roles) != assistants:
        return "greeting_not_identified"
    flags_clear = all(
        _explicit_false(value, name)
        for name in ("streaming", "pending", "failed", "busy", "greetingTruncated")
    )
    greeting = value.get("greeting")
    parts = greeting.split("\n") if isinstance(greeting, str) else []
    if (
        not flags_clear
        or not isinstance(greeting, str)
        or len(parts) != assistants
        or any(part == "" for part in parts)
    ):
        return "greeting_incomplete"
    return None


def greeting_view(value: dict[str, Any]) -> tuple[object, ...]:
    """Identity of one greeting observation, used to detect a change."""

    return (
        value.get("greeting"),
        tuple(as_list(value.get("roles"))),
        value.get("assistantCount"),
        value.get("msgCount"),
        value.get("userCount"),
        value.get("userRows"),
        value.get("transcript"),
        value.get("streaming"),
        value.get("pending"),
        value.get("failed"),
        value.get("busy"),
    )


def file_input_census_expression() -> str:
    """Read the composer file input. This expression does not choose a file."""

    selector = json.dumps(FILE_INPUT_SELECTOR)
    return (
        "(() => { /* grok-desktop:file-input */"
        f" const nodes = Array.from(document.querySelectorAll({selector}));"
        " if (nodes.length !== 1) return {ok:false, count: nodes.length};"
        " const input = nodes[0];"
        " return {ok:true, count: 1, multiple: input.multiple === true,"
        " accept: String(input.getAttribute('accept') || ''),"
        " disabled: input.disabled === true}; })()"
    )


def composer_files_expression() -> str:
    """Read the file input and each composer chip's accessible name.

    The installed composer copies the chosen files and then clears the input,
    so ``input.files`` is not acceptance. An image chip keeps the remembered
    filename in ``aria-label`` and can have empty text.
    """

    selector = json.dumps(FILE_INPUT_SELECTOR)
    return (
        "(() => { /* grok-desktop:composer-files */"
        f" const input = document.querySelector({selector});"
        " if (!input) return {ok:false, reason:'file-input-missing'};"
        " const files = input.files ? Array.from(input.files).map(file => file.name) : [];"
        " const nodes = Array.from(document.querySelectorAll('.sand-prompt-attachment'));"
        " const chips = nodes.map(node => String(node.textContent || '').replace(/\\s+/g, ' ').trim());"
        " const labels = nodes.map(node => String(node.getAttribute('aria-label') || '')"
        ".replace(/\\s+/g, ' ').trim());"
        " return {ok:true, fileCount: files.length, files: files, chips: chips, labels: labels}; })()"
    )


def select_expression(agent_id: str) -> str:
    quoted = json.dumps(agent_id)
    return (
        "(() => { /* grok-desktop:select */"
        f" const id = {quoted};"
        " const buttons = Array.from(document.querySelectorAll('button[data-agent-id]'))"
        ".filter(b => b.getAttribute('data-agent-id') === id);"
        " if (buttons.length !== 1) return {ok:false, reason:'missing', count: buttons.length};"
        " buttons[0].click(); return {ok:true}; })()"
    )


def preflight_expression(agent_id: str, bot_name: str, request_id: str) -> str:
    agent = json.dumps(agent_id)
    name = json.dumps(bot_name)
    tag = json.dumps(request_id)
    return (
        "(() => { /* grok-desktop:preflight */"
        f" const id = {agent}; const expected = {name}; const tag = {tag};"
        " const b = document.querySelector('button[data-agent-id=\"' + id + '\"]');"
        " if (!b || b.getAttribute('data-active') !== 'true') return {ok:false, active:false, nameMatch:false};"
        " function rowName(node) { const desc = String(node.getAttribute('aria-description') || '');"
        " const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT); let n;"
        " while ((n = walker.nextNode())) { const t = String(n.textContent || '').replace(/\\s+/g, ' ').trim();"
        " if (!t) continue; if (desc && (desc === t || desc.indexOf(t) === 0)) continue; return t; } return ''; }"
        " const field = document.querySelector('.sand-prompt-field');"
        " const composerEmpty = !field || String(field.textContent || '').trim().length === 0;"
        " const msgs = Array.from(document.querySelectorAll('.sand-message[data-role]')).filter(el =>"
        " !el.hasAttribute('data-activity') && !el.hasAttribute('data-grok-state')"
        " && !el.classList.contains('sand-activity-mark'));"
        " const users = msgs.filter(el => el.getAttribute('data-role') === 'user');"
        " const assistants = msgs.filter(el => el.getAttribute('data-role') === 'assistant');"
        " const tagged = msgs.some(el => String(el.textContent || '').indexOf(tag) >= 0);"
        " const transcriptRows = Array.from(document.querySelectorAll('.sand-transcript-row'));"
        " const userRows = transcriptRows.filter(el => el.getAttribute('data-role') === 'user').length;"
        " const rowOf = (el) => el.closest('.sand-transcript-row');"
        " const pending = msgs.some(el => el.hasAttribute('data-pending')"
        " || (rowOf(el) && rowOf(el).hasAttribute('data-pending')));"
        " const failed = msgs.some(el => el.hasAttribute('data-failed')"
        " || (rowOf(el) && rowOf(el).hasAttribute('data-failed')));"
        " const busy = msgs.some(el => el.getAttribute('aria-busy') === 'true'"
        " || (rowOf(el) && rowOf(el).getAttribute('aria-busy') === 'true'));"
        " const streaming = assistants.some(el => el.hasAttribute('aria-label'));"
        " let greeting = assistants.map(el => String(el.textContent || '').replace(/\\s+/g, ' ').trim()).join('\\n');"
        " let greetingTruncated = false;"
        " if (greeting.length > 20000) { greeting = greeting.slice(0, 20000); greetingTruncated = true; }"
        " const taggedRow = transcriptRows.some(el => String(el.textContent || '').indexOf(tag) >= 0);"
        " const mounted = transcriptRows.filter(el => el.querySelector('.sand-message[data-role]')).length;"
        " let transcript = 'ready';"
        " if (taggedRow && !tagged) transcript = 'virtualized';"
        " else if (mounted !== msgs.length) transcript = 'ambiguous';"
        " const raw = document.querySelector('[data-grok-state]');"
        " const st = raw ? raw.getAttribute('data-grok-state') : 'absent';"
        " return {active:true, nameMatch: rowName(b) === expected, composerEmpty: composerEmpty,"
        " userCount: users.length, userRows: userRows, assistantCount: assistants.length,"
        " msgCount: msgs.length, tagged: tagged, state: st, transcript: transcript,"
        " pending: pending, failed: failed, busy: busy, streaming: streaming,"
        " greetingTruncated: greetingTruncated, greeting: greeting,"
        " roles: msgs.map(el => el.getAttribute('data-role') || '')}; })()"
    )


def focus_expression() -> str:
    return (
        "(() => { /* grok-desktop:focus */"
        " const field = document.querySelector('.sand-prompt-field');"
        " if (!field) return {ok:false}; field.focus();"
        " return {ok: document.activeElement === field || field.contains(document.activeElement)}; })()"
    )


def submit_expression(request_id: str) -> str:
    tag = json.dumps(request_id)
    return (
        "(() => { /* grok-desktop:submit */"
        f" const tag = {tag};"
        " const field = document.querySelector('.sand-prompt-field');"
        " const text = field ? String(field.textContent || '') : '';"
        " if (text.indexOf(tag) < 0) return {ok:false, reason:'composer-missing-tag'};"
        " const form = field.closest('form');"
        " const button = form ? form.querySelector('button[type=\"submit\"]') : null;"
        " if (!button || button.disabled) return {ok:false, reason:'no-submit'};"
        " button.click(); return {ok:true}; })()"
    )


def _js_tag_map(tags: frozenset[str]) -> str:
    return "{" + ",".join(f"{tag.upper()}:1" for tag in sorted(tags)) + "}"


def poll_expression(agent_id: str, request_id: str) -> str:
    agent = json.dumps(agent_id)
    tag = json.dumps(request_id)
    return (
        "(() => { /* grok-desktop:poll */"
        f" const tag = {tag}; const agentId = {agent}; const limit = {MAX_REPLY};"
        " const active = document.querySelector('button[data-agent-id][data-active=\"true\"]');"
        " const activeAgent = active ? String(active.getAttribute('data-agent-id') || '') : '';"
        " const msgs = Array.from(document.querySelectorAll('.sand-message[data-role]')).filter(el =>"
        " !el.hasAttribute('data-activity') && !el.classList.contains('sand-activity-mark'));"
        " const user = msgs.filter(el => el.getAttribute('data-role') === 'user'"
        " && String(el.textContent || '').indexOf(tag) >= 0);"
        " const rowHit = Array.from(document.querySelectorAll('.sand-transcript-row')).some(el =>"
        " String(el.textContent || '').indexOf(tag) >= 0);"
        " let attribution = 'missing';"
        " if (user.length > 1) attribution = 'ambiguous';"
        " else if (user.length === 1) attribution = 'turn';"
        " else if (rowHit) attribution = 'virtualized';"
        " const all = Array.from(document.querySelectorAll('[data-grok-state], [data-activity]'));"
        " const stateValues = (el) => { const values = [];"
        " const grok = el.getAttribute('data-grok-state');"
        " const activity = el.getAttribute('data-activity');"
        " if (grok) values.push(grok);"
        " if (activity && activity !== grok) values.push(activity);"
        " return values; };"
        " const chrome = (el) => !el || el.nodeType !== 1 ? false : ("
        " el.hasAttribute('data-activity') || el.hasAttribute('data-grok-state') || el.hasAttribute('data-typing')"
        " || el.classList.contains('sand-activity-mark') || el.classList.contains('sand-activity-line')"
        " || el.classList.contains('sand-message-typing') || el.classList.contains('sand-message-attachments')"
        " || el.tagName === 'TIME' || el.hasAttribute('hidden'));"
        f" const INLINE = {_js_tag_map(_INLINE_TAGS)};"
        f" const PARA = {_js_tag_map(_PARAGRAPH_TAGS)};"
        f" const WRAP = {_js_tag_map(_WRAPPER_TAGS)};"
        " const inlineText = (node) => { let out = ''; for (const child of Array.from(node.childNodes)) {"
        " if (child.nodeType === 3) { out += child.nodeValue || ''; continue; }"
        " if (child.nodeType !== 1 || chrome(child)) continue;"
        " if (child.tagName === 'BR') { out += '\\n'; continue; }"
        " if (INLINE[child.tagName]) { const inner = inlineText(child); if (inner == null) return null; out += inner; continue; }"
        " return null; } return out; };"
        " const blocksOf = (node) => { const kids = Array.from(node.childNodes);"
        " const hasBlock = kids.some(child => child.nodeType === 1 && !chrome(child) &&"
        " (PARA[child.tagName] || child.tagName === 'DIV' || WRAP[child.tagName] || child.tagName === 'TABLE'));"
        " const blocks = []; let inline = ''; let saw = false; const flush = () => {"
        " if (!saw) return true; const text = inline; inline = ''; saw = false;"
        " if (hasBlock && text.trim() === '') return text.indexOf('\\n') < 0;"
        " if (hasBlock) return false; blocks.push(text); return true; };"
        " for (const child of kids) {"
        " if (child.nodeType === 3) { const value = child.nodeValue || '';"
        " if (hasBlock && value.trim() === '') continue; inline += value; saw = true; continue; }"
        " if (child.nodeType !== 1 || chrome(child)) continue;"
        " if (child.tagName === 'BR') { inline += '\\n'; saw = true; continue; }"
        " if (INLINE[child.tagName]) { const inner = inlineText(child); if (inner == null) return null;"
        " inline += inner; saw = true; continue; }"
        " if (child.tagName === 'TABLE') return null;"
        " if (WRAP[child.tagName] || child.tagName === 'DIV') { if (!flush()) return null;"
        " const inner = blocksOf(child); if (inner == null) return null;"
        " const transparent = child.tagName !== 'DIV' || Array.from(child.childNodes).some(item =>"
        " item.nodeType === 1 && !chrome(item) && (PARA[item.tagName] || item.tagName === 'DIV'"
        " || WRAP[item.tagName] || item.tagName === 'TABLE'));"
        " if (transparent) inner.forEach(part => blocks.push(part));"
        " else if (inner.length === 1) blocks.push(inner[0]);"
        " else if (inner.length > 1) return null; continue; }"
        " if (PARA[child.tagName]) { if (!flush()) return null;"
        " if (child.tagName === 'HR') { blocks.push(''); continue; }"
        " const text = inlineText(child); if (text == null) return null; blocks.push(text); continue; }"
        " return null; } if (!flush()) return null; return blocks; };"
        " const proseText = (root) => { const blocks = blocksOf(root); return blocks == null ? null : blocks.join('\\n\\n'); };"
        " let states = []; let unrelated = []; let reply = ''; let truncated = false;"
        " let replyChars = 0; let replyStructure = 'absent';"
        " let streaming = false; let pending = false; let failed = false; let busy = false;"
        " if (user.length === 1) {"
        " const start = user[0];"
        " const row = start.closest('.sand-transcript-row');"
        " const rows = [];"
        " if (!row) attribution = 'ambiguous';"
        " else {"
        " rows.push(row);"
        " let sib = row.nextElementSibling;"
        " while (sib) {"
        " const isRow = !!(sib.classList && sib.classList.contains('sand-transcript-row'));"
        " const isInset = !!(sib.classList && sib.classList.contains("
        " 'sand-virtual-transcript__inset'));"
        " if (!isRow && isInset) {"
        " let scan = sib.nextElementSibling; let laterRow = false;"
        " while (scan) {"
        " if (scan.classList && scan.classList.contains('sand-transcript-row')) {"
        " laterRow = true; break; }"
        " scan = scan.nextElementSibling; }"
        " if (laterRow) { attribution = 'ambiguous'; rows.length = 0; }"
        " break; }"
        " if (!isRow) { attribution = 'ambiguous'; rows.length = 0; break; }"
        " if (sib.getAttribute('data-role') === 'user') break;"
        " rows.push(sib); sib = sib.nextElementSibling; } }"
        " const inside = (el) => rows.some(node => node === el || node.contains(el));"
        " pending = rows.some(node => node.hasAttribute('data-pending'));"
        " failed = rows.some(node => node.hasAttribute('data-failed'));"
        " busy = rows.some(node => node.getAttribute('aria-busy') === 'true');"
        " states = all.filter(inside).flatMap(stateValues).slice(0, 8);"
        " unrelated = all.filter(el => !inside(el)).flatMap(stateValues).slice(0, 8);"
        " const assistants = msgs.filter(el => inside(el) && el.getAttribute('data-role') === 'assistant'"
        " && !el.hasAttribute('data-grok-state') && !el.classList.contains('sand-activity-line')"
        " && !el.classList.contains('sand-activity-mark') && !el.hasAttribute('data-activity'));"
        " streaming = assistants.some(el => el.hasAttribute('aria-label'));"
        " if (assistants.length > 1) replyStructure = 'inconclusive';"
        " else if (assistants.length === 1) {"
        " const prose = Array.from(assistants[0].querySelectorAll('.sand-message-prose')).filter(el => !chrome(el));"
        " if (prose.length !== 1) replyStructure = 'inconclusive';"
        " else { const full = proseText(prose[0]);"
        " if (full == null) replyStructure = 'inconclusive';"
        " else { replyChars = full.length; replyStructure = 'prose';"
        " if (full.length > limit) truncated = true; else reply = full; } } } }"
        " else unrelated = all.flatMap(stateValues).slice(0, 8);"
        " return {userHits: user.length, activeAgent: activeAgent, agentMatch: activeAgent === agentId,"
        " attribution: attribution, replyStructure: replyStructure,"
        " states: states, unrelatedStates: unrelated, reply: reply, truncated: truncated,"
        " replyChars: replyChars, streaming: streaming, pending: pending, failed: failed, busy: busy}; })()"
    )
