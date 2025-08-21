import sys
import json
import logging
import re
import bisect
import re
import time
from jmcc_extension import try_find_object
from pathlib import Path
from urllib.parse import urlparse, unquote
import jmcc_extension
from jmcc_extension import Tokens

DEBOUNCE_INTERVAL = 0.1

logging.basicConfig(
    level=logging.INFO,
    format="[LSP] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)

def log(msg: str):
    logging.info(msg)

def read_message():
    try:
        headers = {}
        while True:
            line = sys.stdin.buffer.readline().decode("utf-8", errors="replace")
            if not line or line.strip() == "":
                break
            key, val = line.split(":", 1)
            headers[key.strip()] = val.strip()
        length = int(headers.get("Content-Length", 0))
        if length == 0:
            return None
        body = sys.stdin.buffer.read(length).decode("utf-8")
        return json.loads(body)
    except Exception as e:
        log(f"Error reading message: {e}")
        return None

def send_message(msg: dict):
    try:
        body = json.dumps(msg, ensure_ascii=False).encode("utf-8")
        header = (
            f"Content-Length: {len(body)}\r\n"
            "Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n"
            "\r\n"
        ).encode("utf-8")
        sys.stdout.buffer.write(header + body)
        sys.stdout.buffer.flush()
    except Exception as e:
        log(f"Error sending message: {e}")


symbol_items = []
symbol_index = {}
symbol_labels = []
document_states = {}
file_cache = {}

def uri_to_path(uri: str) -> Path | None:
    p = urlparse(uri)
    if p.scheme != "file":
        return None
    path = unquote(p.path)
    if sys.platform.startswith("win") and path.startswith("/"):
        path = path[1:]
    return Path(path)

def path_to_uri(path: Path) -> str:
    p = path.resolve().as_posix()
    if sys.platform.startswith("win"):
        p = "/" + p
    return "file://" + p

def read_document(uri: str) -> str | None:
    path = uri_to_path(uri)
    if not path or not path.is_file():
        return None
    mtime = path.stat().st_mtime
    cached = file_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    text = path.read_text(encoding="utf-8").replace("\r", "")
    file_cache[path] = (mtime, text)
    return text

def load_symbols():
    global symbol_items, symbol_index, symbol_labels
    if symbol_items:
        return
    assets = Path(__file__).parent / "assets" / "completions.json"
    data = json.loads(assets.read_text(encoding="utf-8"))
    symbol_items = data
    for item in data:
        label = item.get("label", "")
        clean = re.sub(r"\(\)$", "", label)
        symbol_index[clean] = item
    symbol_labels = sorted(symbol_index.keys())
    log(f"Loaded {len(symbol_labels)} symbols")

def get_completions(prefix: str) -> list[dict]:
    if not prefix:
        return [it.copy() for it in symbol_items]
    left = bisect.bisect_left(symbol_labels, prefix)
    high = prefix + "\uffff"
    right = bisect.bisect_right(symbol_labels, high)
    return [symbol_index[symbol_labels[i]].copy() for i in range(left, right)]

def compute_line_offsets(text: str) -> list[int]:
    offsets = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            offsets.append(idx + 1)
    return offsets

def ensure_tokens(uri: str):
    state = document_states[uri]
    now = time.time()
    last = state.get("last_token_time", 0.0)
    if state["dirty"] and (now - last) >= DEBOUNCE_INTERVAL:
        jmcc_extension.clear(uri)
        tokens = jmcc_extension.tokenize(state["text"], uri, True)
        starts = [t.starting_pos for t in tokens]
        offsets = compute_line_offsets(state["text"])
        state.update({
            "tokens": tokens,
            "start_positions": starts,
            "line_offsets": offsets,
            "dirty": False,
            "last_token_time": now
        })
        log(f"Tokenized {uri}: {len(tokens)} tokens")
    return state

def line_col_to_pos(line: int, col: int, offsets: list[int]) -> int:
    return offsets[line] + col

def pos_to_line_col(pos: int, offsets: list[int]) -> tuple[int,int]:
    line = bisect.bisect_right(offsets, pos) - 1
    col = pos - offsets[line]
    return line, col

_def_pattern = re.compile(
    r"\b(?:class|function|process|var|def|fun)\b.*?\b(\w+)\b|"
    r"class\s+(\w+)\s*{"
)

def build_definitions(uri: str):
    state = document_states[uri]
    defs = {}
    for i, line in enumerate(state["text"].splitlines()):
        clean = line.split("//", 1)[0]
        m = _def_pattern.search(clean)
        if m:
            name = m.group(1) or m.group(2)
            col = clean.find(name)
            defs[name] = {
                "uri": uri,
                "range": {
                    "start": {"line": i, "character": col},
                    "end":   {"line": i, "character": col + len(name)}
                }
            }
    state["definitions"] = defs
    state["defs_built"] = True
    log(f"Built definitions for {uri}: {len(defs)} entries")

def find_definition_in_state(uri: str, word: str):
    if uri not in document_states:
        text = read_document(uri)
        if text is None:
            return None

        document_states[uri] = {
            "text": text,
            "dirty": False,
            "tokens": [],
            "start_positions": [],
            "line_offsets": [],
            "definitions": {},
            "defs_built": False
        }

    state = document_states[uri]

    if not state["defs_built"]:
        build_definitions(uri)
    return state["definitions"].get(word)

def extract_imports(uri: str, content: str) -> list[str]:
    results = []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("import ") and '"' in line:
            try:
                path = line.split('"',2)[1]
                results.append(path)
            except:
                pass
    return results

def resolve_import_uri(from_uri: str, import_path: str) -> str | None:
    base = uri_to_path(from_uri)
    if not base:
        return None
    candidate = (base.parent / import_path).resolve()
    if not candidate.is_file():
        return None
    return path_to_uri(candidate)

def parse_function_signature(line: str):
    clean = line.split("//",1)[0]
    m = re.search(
        r"\b(?:function|process|def|fun)\b\s+(\w+)\s*\(\s*([^)]*)\s*\)\s*(?:->\s*(\w+))?",
        clean, re.IGNORECASE
    )
    if not m:
        return None
    name = m.group(1)
    raw = m.group(2) or ""
    params = []
    for part in raw.split(","):
        p = re.split(r"[:=]", part.strip())[0].strip()
        if p:
            params.append(p)
    return {"name": name, "params": params}

def get_signature_from_definition(start_uri: str, func_name: str):
    visited = set()
    def _search(uri):
        if uri in visited:
            return None
        visited.add(uri)
        text = document_states.get(uri,{}).get("text") or read_document(uri) or ""
        for line in text.splitlines():
            sig = parse_function_signature(line)
            if sig and sig["name"] == func_name:
                return sig
        for imp in extract_imports(uri, text):
            sub = resolve_import_uri(uri, imp)
            if sub:
                res = _search(sub)
                if res:
                    return res
        return None
    return _search(start_uri)


def handle_initialize(msg):
    global HIDE_INLAY_HINTS
    global HIDE_HOVER
    global HIDE_COMPLETION
    global HIDE_SIGNATURE_HELP

    rpc_id = msg["id"]
    params = msg.get("params", {})
    init_opts = params.get("initializationOptions", {})
    HIDE_INLAY_HINTS            = init_opts.get("hideInlayHints", False)
    HIDE_HOVER                  = init_opts.get("hideHover", False)
    HIDE_COMPLETION             = init_opts.get("hideCompletion", False)
    HIDE_SIGNATURE_HELP         = init_opts.get("hideSignatureHelp", False)
    load_symbols()
    capabilities_dict = {
        "textDocumentSync": {"openClose": True, "change": 1, "save": True},
        "completionProvider": {
            "resolveProvider": False,
            "triggerCharacters": [":", "=", "<", "."]
        } if not HIDE_COMPLETION else None,
        "hoverProvider": not HIDE_HOVER,
        "definitionProvider": True,
        "signatureHelpProvider": {
            "triggerCharacters": ["(", ","],
            "retriggerCharacters": [")"]
        } if not HIDE_SIGNATURE_HELP else None,
        "inlayHintProvider": not HIDE_INLAY_HINTS
    }
    capabilities_dict = {k: v for k, v in capabilities_dict.items() if v is not None}
    send_message({
        "id": rpc_id,
        "result": {
            "capabilities": capabilities_dict
        }
    })

def handle_did_change_configuration(msg):
    cfg = msg.get("params", {})\
             .get("settings", {})\
             .get("jmcc-helper", {})

    global HIDE_INLAY_HINTS, HIDE_HOVER, HIDE_COMPLETION, HIDE_SIGNATURE_HELP

    HIDE_INLAY_HINTS    = cfg.get("hideInlayHints",    HIDE_INLAY_HINTS)
    HIDE_HOVER          = cfg.get("hideHover",         HIDE_HOVER)
    HIDE_COMPLETION     = cfg.get("hideCompletion",    HIDE_COMPLETION)
    HIDE_SIGNATURE_HELP = cfg.get("hideSignatureHelp", HIDE_SIGNATURE_HELP)

    log(f"Config updated: inlay={HIDE_INLAY_HINTS} hover={HIDE_HOVER} "
        f"completion={HIDE_COMPLETION} signature={HIDE_SIGNATURE_HELP}")

def handle_did_open(msg):
    uri = msg["params"]["textDocument"]["uri"]
    text = msg["params"]["textDocument"]["text"].replace("\r", "")
    document_states[uri] = {
        "text": text,
        "dirty": True,
        "tokens": [],
        "start_positions": [],
        "line_offsets": [],
        "definitions": {},
        "defs_built": False,
        "last_token_time": 0.0,
        "inlay_dirty": True,
        "inlay_hints": []
    }
    jmcc_extension.clear(uri)
    log(f"Opened {uri}")

def handle_did_change(msg):
    uri = msg["params"]["textDocument"]["uri"]
    text = msg["params"]["contentChanges"][0]["text"].replace("\r", "")
    state = document_states.get(uri)
    if state:
        state["text"] = text
        state["dirty"] = True
        state["defs_built"] = False
        state["inlay_dirty"] = True
        jmcc_extension.clear(uri)
        log(f"Changed {uri}")

def handle_completion(msg):
    rpc_id = msg["id"]
    uri = msg["params"]["textDocument"]["uri"]
    if HIDE_COMPLETION:
        send_message({"id": rpc_id, "result": []})
        return
    pos = msg["params"]["position"]
    state = document_states.get(uri)
    if not state:
        send_message({"id": rpc_id, "result": {"isIncomplete":False,"items":[]}})
        return

    lines = state["text"].splitlines()
    if pos["line"] >= len(lines):
        send_message({"id": rpc_id, "result": {"isIncomplete":False,"items":[]}})
        return

    line = lines[pos["line"]]
    prefix = re.search(r"[\w:<]*$", line[:pos["character"]]).group(0)
    items = get_completions(prefix)

    edits = []
    for it in items:
        lbl = it["label"]
        start_ch = max(0, pos["character"] - len(prefix))
        edits.append({
            **it,
            "textEdit": {
                "range": {
                    "start": {"line":pos["line"],"character":start_ch},
                    "end":   {"line":pos["line"],"character":pos["character"]}
                },
                "newText": lbl
            }
        })

    send_message({"id":rpc_id,"result":{"isIncomplete":False,"items":edits}})

file_class_index: dict[str, dict[str, list[str]]] = {}

def update_class_index(uri: str, text: str) -> None:
    classes: dict[str, list[str]] = {}
    for match in re.finditer(r'\bclass\s+(\w+)\s*\{', text):
        class_name = match.group(1)
        start = match.end() - 1
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    body = text[start+1:i]
                    methods = re.findall(r'\bfunction\s+(\w+)\s*\(', body)
                    classes[class_name] = methods
                    break

    file_class_index[uri] = classes

def handle_hover(msg):
    rpc_id = msg["id"]
    uri    = msg["params"]["textDocument"]["uri"]
    if HIDE_HOVER:
        send_message({"id": rpc_id, "result": []})
        return
    pos    = msg["params"]["position"]

    try:
        state = ensure_tokens(uri)
        text   = state["text"]
        offsets = state["line_offsets"]
        prev_index = file_class_index.get(uri)
        if prev_index is None or prev_index.get("_text_hash") != hash(text):
            update_class_index(uri, text)
            file_class_index[uri]["_text_hash"] = hash(text)

        global_pos = line_col_to_pos(pos["line"], pos["character"], offsets)
        idx = bisect.bisect_right(state["start_positions"], global_pos) - 1
        if idx < 0 or idx >= len(state["tokens"]):
            return send_message({"id": rpc_id, "result": None})

        tok  = state["tokens"][idx]
        word = try_find_object(uri, idx)
        if not word:
            return send_message({"id": rpc_id, "result": None})

        contents = None

        item = symbol_index.get(word)
        if item and "documentation" in item:
            doc = item["documentation"]
            contents = doc if isinstance(doc, dict) else {"kind": "markdown", "value": str(doc)}

        elif sig_def := get_signature_from_definition(uri, word):
            params = sig_def.get("params", [])
            md = "`(" + ", ".join(params) + ")`"
            contents = {"kind": "markdown", "value": md}

        else:
            cls_methods = file_class_index.get(uri, {}).get(word)
            if cls_methods:
                md = f"**Методы класса `{word}`**:\n\n"
                md += "\n".join(f"- `{m}()`" for m in sorted(cls_methods))
                contents = {"kind": "markdown", "value": md}

        if contents is None:
            return send_message({"id": rpc_id, "result": None})

        sl, sc = pos_to_line_col(tok.starting_pos, offsets)
        el, ec = pos_to_line_col(tok.ending_pos,   offsets)

        send_message({
            "id": rpc_id,
            "result": {
                "contents": contents,
                "range": {
                    "start": {"line": sl, "character": sc},
                    "end":   {"line": el, "character": ec}
                }
            }
        })

    except Exception as e:
        print("Ошибка в handle_hover:", e)
        send_message({"id": rpc_id, "result": None})

def handle_definition(msg):
    rpc_id = msg["id"]
    uri = msg["params"]["textDocument"]["uri"]
    pos = msg["params"]["position"]
    state = document_states.get(uri)
    if not state:
        send_message({"id":rpc_id,"result":None})
        return

    lines = state["text"].splitlines()
    if pos["line"] >= len(lines):
        send_message({"id":rpc_id,"result":None})
        return

    line = lines[pos["line"]]
    word = None
    for m in re.finditer(r"[\w_]+", line):
        if m.start() <= pos["character"] <= m.end():
            word = m.group(0)
            break
    if not word:
        send_message({"id":rpc_id,"result":None})
        return

    loc = find_definition_in_state(uri, word)
    if loc:
        send_message({"id":rpc_id,"result":loc})
        return

    for imp in extract_imports(uri, state["text"]):
        iuri = resolve_import_uri(uri, imp)
        if iuri:
            loc = find_definition_in_state(iuri, word)
            if loc:
                send_message({"id":rpc_id,"result":loc})
                return

    send_message({"id":rpc_id,"result":None})

def handle_signature_help(msg):
    rpc_id = msg["id"]
    uri = msg["params"]["textDocument"]["uri"]
    if HIDE_SIGNATURE_HELP:
        send_message({"id": rpc_id, "result": []})
        return
    pos = msg["params"]["position"]
    state = document_states.get(uri)
    if not state:
        send_message({"id":rpc_id,"result":None})
        return

    lines = state["text"].splitlines()
    if pos["line"] >= len(lines):
        send_message({"id":rpc_id,"result":None})
        return

    line = lines[pos["line"]]
    char = pos["character"]

    bracket = -1
    depth = 0
    for i in range(char-1, -1, -1):
        c = line[i]
        if c == ")":
            depth += 1
        elif c == "(" and depth == 0:
            bracket = i
            break
        elif c == "(":
            depth -= 1
    if bracket < 0:
        send_message({"id":rpc_id,"result":None})
        return

    start = bracket-1
    while start >= 0 and (line[start].isalnum() or line[start] in "_:<"):
        start -= 1
    func = line[start+1:bracket].strip()
    if not func:
        send_message({"id":rpc_id,"result":None})
        return

    item = symbol_index.get(func)
    sig_data = None
    if item and "signature" in item:
        sig_data = item["signature"]
    else:
        sig_def = get_signature_from_definition(uri, func)
        if sig_def:
            sig_data = {
                "label": f"{func}({', '.join(sig_def['params'])})",
                "parameters": [{"label": p} for p in sig_def["params"]],
                "documentation": ""
            }
    if not sig_data:
        send_message({"id":rpc_id,"result":None})
        return

    inner = line[bracket+1:char]
    active = inner.count(",")

    send_message({
        "id": rpc_id,
        "result": {
            "signatures": [
                {
                    "label": sig_data.get("label", f"{func}(...)"),
                    "documentation": sig_data.get("documentation", ""),
                    "parameters": [
                        {"label": p.get("label", p["label"]), "documentation": p.get("documentation","")}
                        for p in sig_data.get("parameters", [])
                    ]
                }
            ],
            "activeSignature": 0,
            "activeParameter": active
        }
    })

def handle_inlay_hints(msg):
    rpc_id = msg["id"]
    uri = msg["params"]["textDocument"]["uri"]

    if HIDE_INLAY_HINTS:
        send_message({"id": rpc_id, "result": []})
        return

    state = ensure_tokens(uri)
    if not state["inlay_dirty"]:
        send_message({"id": rpc_id, "result": state["inlay_hints"]})
        return

    tokens = state["tokens"]
    offsets = state["line_offsets"]
    lines = state["text"].splitlines()
    hints = []
    idx = 0

    def load_signature(func_name: str, is_method: bool = False) -> list[str]:
        item = symbol_index.get(func_name) or (symbol_index.get("." + func_name) if not func_name.startswith(".") else None)
        if item and "signature" in item:
            return [p["label"].split(":")[0].strip() for p in item["signature"]["parameters"]]
        if (is_method and "." in func_name) or (not is_method and "::" in func_name):
            sep = "." if is_method else "::"
            bare = func_name.split(sep, 1)[1]
            item2 = symbol_index.get(bare) or (symbol_index.get("." + bare) if not bare.startswith(".") else None)
            if item2 and "signature" in item2:
                return [p["label"].split(":")[0].strip() for p in item2["signature"]["parameters"]]
        sig_def = get_signature_from_definition(uri, func_name)
        if not sig_def:
            if is_method and "." in func_name:
                sig_def = get_signature_from_definition(uri, func_name.split(".",1)[1])
            elif not is_method and "::" in func_name:
                sig_def = get_signature_from_definition(uri, func_name.split("::",1)[1])
        return sig_def.get("params", []) if sig_def else []
    def is_definition_header(pos: int) -> bool:
        j = pos - 1
        while j >= 0 and tokens[j].type in (Tokens.NEXT_LINE, Tokens.SEMICOLON, Tokens.COLON):
            j -= 1
        return j >= 0 and tokens[j].type in (
            Tokens.FUNCTION_DEFINE, Tokens.PROCESS_DEFINE, Tokens.CLASS_DEFINE
        )

    while idx < len(tokens):
        t = tokens[idx]
        func_name = None
        open_paren = None
        static_call = False
        method_call = False
        if (
            t.type == Tokens.VARIABLE and
            idx + 1 < len(tokens) and tokens[idx+1].type == Tokens.LPAREN and
            not is_definition_header(idx)
        ):
            func_name = t.value
            open_paren = idx + 1
        elif (
            t.type == Tokens.VARIABLE and
            idx + 1 < len(tokens) and tokens[idx+1].type == Tokens.DOUBLE_COLON and
            idx + 2 < len(tokens) and tokens[idx+2].type == Tokens.VARIABLE and
            idx + 3 < len(tokens) and tokens[idx+3].type == Tokens.LPAREN and
            not is_definition_header(idx)
        ):
            func_name = f"{t.value}::{tokens[idx+2].value}"
            open_paren = idx + 3
            static_call = True
        elif t.type == Tokens.VARIABLE:
            j = idx+1
            parts = [t.value]
            while j+1 < len(tokens) and tokens[j].type == Tokens.DOT and tokens[j+1].type == Tokens.VARIABLE:
                parts.append(tokens[j+1].value)
                j += 2
            if len(parts) > 1 and j < len(tokens) and tokens[j].type == Tokens.LPAREN and not is_definition_header(idx):
                func_name = ".".join(parts)
                open_paren = j
                method_call = True

        if open_paren is None:
            idx += 1
            continue
        depth = 0
        close_paren = None
        for j in range(open_paren, len(tokens)):
            tp = tokens[j].type
            if tp in (Tokens.LPAREN, Tokens.LSPAREN, Tokens.LCPAREN):
                depth += 1
            elif tp in (Tokens.RPAREN, Tokens.RSPAREN, Tokens.RCPAREN):
                depth -= 1
                if depth == 0:
                    close_paren = j
                    break
        if close_paren is None:
            idx = open_paren + 1
            continue

        sig_params = load_signature(func_name, is_method=method_call)
        if not sig_params:
            idx = close_paren + 1
            continue

        args = tokens[open_paren+1 : close_paren]
        if not args:
            idx = close_paren + 1
            continue
        offset = 0
        if func_name.startswith("repeat::"):
            block_start = None
            for k in range(close_paren+1, len(tokens)):
                if tokens[k].type in (Tokens.NEXT_LINE, Tokens.SEMICOLON):
                    continue
                if tokens[k].type == Tokens.LCPAREN:
                    block_start = k
                break
            if block_start is not None:
                bd = 1
                block_end = None
                for k in range(block_start+1, len(tokens)):
                    if tokens[k].type == Tokens.LCPAREN:
                        bd += 1
                    elif tokens[k].type == Tokens.RCPAREN:
                        bd -= 1
                        if bd == 0:
                            block_end = k
                            break
                if block_end is not None:
                    vc = 0
                    for tt in tokens[block_start+1:block_end]:
                        if tt.type == Tokens.CYCLE_THING:
                            break
                        if tt.type == Tokens.VARIABLE:
                            vc += 1
                    offset = vc
        elif static_call or method_call:
            assign_idx = None
            return_found = False
            for j in range(idx-1, -1, -1):
                prev = tokens[j]
                if prev.type in (Tokens.SEMICOLON, Tokens.NEXT_LINE):
                    break
                if prev.type == Tokens.ASSIGN:
                    assign_idx = j
                    break
                if prev.type == Tokens.RETURN:
                    return_found = True
                    break
            if assign_idx is not None and not (method_call and len(sig_params) == 1):
                vc = 0
                k = assign_idx - 1
                while k >= 0 and tokens[k].type in (Tokens.VARIABLE, Tokens.COMMA):
                    if tokens[k].type == Tokens.VARIABLE:
                        vc += 1
                    k -= 1
                offset = vc
            elif return_found and not (method_call and len(sig_params) == 1):
                offset = 1
        active = sig_params[offset:]
        if not active:
            idx = close_paren + 1
            continue
        def split_top(ts):
            groups, cur, lvl = [], [], 0
            for tt in ts:
                if tt.type in (Tokens.LPAREN, Tokens.LSPAREN, Tokens.LCPAREN):
                    lvl += 1
                elif tt.type in (Tokens.RPAREN, Tokens.RSPAREN, Tokens.RCPAREN):
                    lvl -= 1
                if tt.type == Tokens.COMMA and lvl == 0:
                    groups.append(cur)
                    cur = []
                else:
                    cur.append(tt)
            if cur:
                groups.append(cur)
            return groups

        groups = split_top(args)
        named = set()
        for grp in groups:
            for i2, tok2 in enumerate(grp):
                if tok2.type == Tokens.ASSIGN and i2 > 0 and grp[i2-1].type == Tokens.VARIABLE:
                    named.add(grp[i2-1].value)
        active = [p for p in active if p not in named]
        if not active:
            idx = close_paren + 1
            continue
        pidx = 0
        for grp in groups:
            if any(tt.type == Tokens.ASSIGN for tt in grp):
                continue

            if pidx >= len(active):
                break
            first_tok = next((x for x in grp if x.type != Tokens.NEXT_LINE), None)
            if not first_tok:
                continue

            ln, ch = pos_to_line_col(first_tok.starting_pos, offsets)
            hints.append({
                "position": {"line": ln, "character": ch},
                "label": active[pidx] + ":",
                "kind": 1,
                "paddingLeft": False,
                "paddingRight": True
            })
            pidx += 1

        idx = close_paren + 1

    state["inlay_hints"] = hints
    state["inlay_dirty"] = False
    send_message({"id": rpc_id, "result": hints})


def main():
    log("Server starting")
    while True:
        msg = read_message()
        if not msg:
            continue
        method = msg.get("method")
        if method == "initialize":
            handle_initialize(msg)
        elif method == "workspace/didChangeConfiguration":
            handle_did_change_configuration(msg)
        elif method == "textDocument/didOpen":
            handle_did_open(msg)
        elif method == "textDocument/didChange":
            handle_did_change(msg)
        elif method == "textDocument/completion":
            handle_completion(msg)
        elif method == "textDocument/hover":
            handle_hover(msg)
        elif method == "textDocument/definition":
            handle_definition(msg)
        elif method == "textDocument/signatureHelp":
            handle_signature_help(msg)
        elif method == "textDocument/inlayHint":
            handle_inlay_hints(msg)
        elif method == "exit":
            log("Exit requested")
            break
        else:
            if method not in ("textDocument/didSave",):
                log(f"Unknown method: {method}")

if __name__ == "__main__":
    main()