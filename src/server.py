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
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        func = None
        open_idx = None
        static_call = False
        method_call = False
        orig_method_call = False
        block_vars = 0
        drop_first = False
        if tok.type == Tokens.VARIABLE and i + 1 < len(tokens) and tokens[i + 1].type == Tokens.LPAREN:
            func, open_idx = tok.value, i + 1
        elif tok.type == Tokens.VARIABLE and i + 3 < len(tokens) \
             and tokens[i + 1].type == Tokens.DOUBLE_COLON \
             and tokens[i + 2].type == Tokens.VARIABLE \
             and tokens[i + 3].type == Tokens.LPAREN:
            func = f"{tok.value}::{tokens[i + 2].value}"
            open_idx = i + 3
            static_call = True
        elif tok.type == Tokens.VARIABLE and i + 3 < len(tokens) \
             and tokens[i + 1].type == Tokens.DOT \
             and tokens[i + 2].type == Tokens.VARIABLE \
             and tokens[i + 3].type == Tokens.LPAREN:
            raw_method = tokens[i + 2].value
            func = f"{tok.value}.{raw_method}"
            open_idx = i + 3
            method_call = True
            orig_method_call = True
        if open_idx is None:
            i += 1
            continue
        if method_call and func not in symbol_index:
            for full_name in symbol_index:
                if full_name.split("::")[-1] == raw_method:
                    func = full_name
                    static_call = True
                    method_call = False
                    break
        depth = 0
        close = None
        for j in range(open_idx, len(tokens)):
            t = tokens[j]
            if t.type in (Tokens.LPAREN, Tokens.LSPAREN, Tokens.LCPAREN):
                depth += 1
            elif t.type in (Tokens.RPAREN, Tokens.RSPAREN, Tokens.RCPAREN):
                depth -= 1
                if depth == 0:
                    close = j
                    break
        if close is None:
            i = open_idx + 1
            continue
        sl, sc = pos_to_line_col(tokens[open_idx].starting_pos, offsets)
        prefix = lines[sl][:sc]
        sig = parse_function_signature(lines[sl])
        if sig and sig["name"] == func:
            i = close + 1
            continue
        args = tokens[open_idx + 1 : close]
        if not args:
            i = close + 1
            continue
        params = []
        if func in symbol_index and "signature" in symbol_index[func]:
            sig_data = symbol_index[func]["signature"]
            params = [
                p["label"].split(":")[0].strip()
                for p in sig_data.get("parameters", [])
            ]
        else:
            sig_def = get_signature_from_definition(uri, func)
            if sig_def:
                params = sig_def["params"]
        is_repeat_function = func.startswith("repeat::")
        repeat_block_offset = 0
        
        if is_repeat_function:
            j = close + 1
            while j < len(tokens) and tokens[j].type != Tokens.LCPAREN:
                j += 1
            
            if j < len(tokens) and tokens[j].type == Tokens.LCPAREN:
                block_start = j
                block_end = None
                block_depth = 0
                
                for k in range(j, len(tokens)):
                    if tokens[k].type == Tokens.LCPAREN:
                        block_depth += 1
                    elif tokens[k].type == Tokens.RCPAREN:
                        block_depth -= 1
                        if block_depth == 0:
                            block_end = k
                            break
                
                if block_end:
                    for k in range(block_start, block_end):
                        if tokens[k].type == Tokens.CYCLE_THING:
                            vars_before_arrow = 0
                            for l in range(block_start + 1, k):
                                if tokens[l].type == Tokens.VARIABLE:
                                    vars_before_arrow += 1
                                elif tokens[l].type == Tokens.COMMA:
                                    vars_before_arrow += 1
                            
                            if func == "repeat::for_each_in_list":
                                repeat_block_offset = max(0, vars_before_arrow - 2)
                            else:
                                repeat_block_offset = max(0, vars_before_arrow - 1)
                            break
        
        is_multiple_assignment = False
        assigned_variable_count = 0
        temp_search = i - 1
        while temp_search >= 0:
            if tokens[temp_search].type == Tokens.ASSIGN:
                var_count = 0
                temp_var_search = temp_search - 1
                while temp_var_search >= 0:
                    t2 = tokens[temp_var_search]
                    if t2.type == Tokens.COMMA:
                        pass
                    elif t2.type == Tokens.VARIABLE:
                        var_count += 1
                    elif t2.type in (
                        Tokens.LPAREN, Tokens.LSPAREN, Tokens.LCPAREN,
                        Tokens.NEXT_LINE, Tokens.SEMICOLON
                    ):
                        break
                    temp_var_search -= 1
                if var_count > 0:
                    is_multiple_assignment = True
                    assigned_variable_count = var_count
                break
            if tokens[temp_search].type in (Tokens.NEXT_LINE, Tokens.SEMICOLON):
                break
            temp_search -= 1
        has_return = False
        temp_check = i - 1
        while temp_check >= 0:
            if tokens[temp_check].type == Tokens.RETURN:
                has_return = True
                break
            if tokens[temp_check].type in (Tokens.NEXT_LINE, Tokens.SEMICOLON):
                break
            temp_check -= 1
        if orig_method_call or (static_call and '=' in prefix):
            drop_first = True
        if is_repeat_function:
            j = close + 1
            found_block = False
            while j < len(tokens):
                if tokens[j].type == Tokens.LCPAREN:
                    block_start = j
                    block_depth = 0
                    block_end = None
                    for k in range(j, len(tokens)):
                        if tokens[k].type == Tokens.LCPAREN:
                            block_depth += 1
                        elif tokens[k].type == Tokens.RCPAREN:
                            block_depth -= 1
                            if block_depth == 0:
                                block_end = k
                                break
                    
                    if block_end:
                        for k in range(block_start, block_end):
                            if tokens[k].type == Tokens.CYCLE_THING:
                                found_block = True
                                break
                    break
                elif tokens[j].type in (Tokens.RPAREN, Tokens.RSPAREN, Tokens.RCPAREN, Tokens.NEXT_LINE, Tokens.SEMICOLON):
                    break
                j += 1
            if found_block:
                drop_first = True
            else:
                drop_first = False
        if is_multiple_assignment and params:
            has_named = any(t.type == Tokens.ASSIGN for t in args)
            if has_named:
                named = set()
                for k, t3 in enumerate(args):
                    if t3.type == Tokens.ASSIGN and k > 0 and args[k-1].type == Tokens.VARIABLE:
                        named.add(args[k-1].value)
                remaining = [p for p in params if p not in named]
                params = [remaining[-1]] if remaining else []
            else:
                if assigned_variable_count > 0 and len(params) > assigned_variable_count:
                    params = params[assigned_variable_count:]
        else:
            if drop_first and params:
                if len(params) > 1:
                    params = params[1:]
            if block_vars > 0:
                params = params[block_vars:]
        if has_return and params:
            if len(params) > 1:
                params = params[1:]
        if is_repeat_function and repeat_block_offset > 0:
            if len(params) > repeat_block_offset:
                params = params[repeat_block_offset:]
            else:
                params = []
        if not params:
            i = close + 1
            continue
        def split_top(lst):
            out, cur, lvl = [], [], 0
            for t4 in lst:
                if t4.type in (Tokens.LPAREN, Tokens.LSPAREN, Tokens.LCPAREN):
                    lvl += 1
                elif t4.type in (Tokens.RPAREN, Tokens.RSPAREN, Tokens.RCPAREN):
                    lvl -= 1
                if t4.type == Tokens.COMMA and lvl == 0:
                    out.append(cur)
                    cur = []
                else:
                    cur.append(t4)
            if cur:
                out.append(cur)
            return out
        groups = split_top(args)
        used_named = set()
        pidx = 0
        for grp in groups:
            if not grp:
                continue
            if any(t.type == Tokens.ASSIGN for t in grp):
                name = next((t.value for t in grp if t.type == Tokens.VARIABLE), None)
                if name:
                    used_named.add(name)
                continue
            while pidx < len(params) and params[pidx] in used_named:
                pidx += 1
            if pidx >= len(params):
                break
            first_token = None
            for t5 in grp:
                if t5.type != Tokens.NEXT_LINE:
                    first_token = t5
                    break
            if first_token is None:
                continue
            ln, ch = pos_to_line_col(first_token.starting_pos, offsets)
            hint = {
                "position": {"line": ln, "character": ch},
                "label": params[pidx] + ":",
                "kind": 1,
                "paddingLeft": False,
                "paddingRight": True
            }
            hints.append(hint)
            pidx += 1
        i = close + 1
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