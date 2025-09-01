import sys
import json
import logging
import re
import bisect
import time
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Any, Dict, List, Optional, Tuple, Callable

from jmcc_extension import Tokens, tokenize, clear, try_find_object

DEBOUNCE_INTERVAL = 0.10

logging.basicConfig(
    level=logging.INFO,
    format="[LSP] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger("LSP")

def read_message() -> Optional[dict]:
    try:
        headers: Dict[str, str] = {}
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            try:
                s = line.decode("utf-8", errors="replace")
            except Exception:
                s = str(line)
            s = s.strip("\r\n")
            if s == "":
                break
            if ":" in s:
                key, val = s.split(":", 1)
                headers[key.strip().lower()] = val.strip()

        length_s = headers.get("content-length", "0")
        try:
            length = int(length_s)
        except ValueError:
            length = 0

        if length <= 0:
            return {}
        body = sys.stdin.buffer.read(length)
        if not body:
            return None
        try:
            text = body.decode("utf-8", errors="replace")
            return json.loads(text)
        except Exception as e:
            log.error(f"Error decoding/JSON parsing body: {e}")
            return {}
    except Exception as e:
        log.error(f"Error reading message: {e}")
        return None


def send_message(msg: dict) -> None:
    try:
        body = json.dumps(msg, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        header = (
            f"Content-Length: {len(body)}\r\n"
            "Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n"
            "\r\n"
        ).encode("utf-8")
        sys.stdout.buffer.write(header)
        sys.stdout.buffer.write(body)
        sys.stdout.buffer.flush()
    except Exception as e:
        log.error(f"Error sending message: {e}")

def uri_to_path(uri: str) -> Optional[Path]:
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

file_cache: Dict[Path, Tuple[float, str]] = {}

def read_document(uri: str) -> Optional[str]:
    path = uri_to_path(uri)
    if not path or not path.is_file():
        return None
    try:
        mtime = path.stat().st_mtime
        cached = file_cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        text = path.read_text(encoding="utf-8").replace("\r", "")
        file_cache[path] = (mtime, text)
        return text
    except Exception as e:
        log.error(f"read_document error for {uri}: {e}")
        return None

def compute_line_offsets(text: str) -> List[int]:
    offsets = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            offsets.append(i + 1)
    return offsets


def line_col_to_pos(line: int, col: int, offsets: List[int]) -> int:
    if line < 0:
        return 0
    if line >= len(offsets):
        return len(offsets) and offsets[-1] or 0
    return offsets[line] + max(0, col)


def pos_to_line_col(pos: int, offsets: List[int]) -> Tuple[int, int]:
    line = bisect.bisect_right(offsets, max(0, pos)) - 1
    line = max(0, min(line, len(offsets) - 1))
    col = pos - offsets[line]
    return line, col

document_states: Dict[str, dict] = {}

symbol_items: List[dict] = []
symbol_index: Dict[str, dict] = {}
symbol_labels: List[str] = []

file_class_index: Dict[str, Dict[str, List[str]]] = {}

HIDE_INLAY_HINTS = False
HIDE_HOVER = False
HIDE_COMPLETION = False
HIDE_SIGNATURE_HELP = False


def load_symbols() -> None:
    global symbol_items, symbol_index, symbol_labels
    try:
        assets_path = (Path(__file__).parent / "assets" / "completions.json").resolve()
        if not assets_path.is_file():
            symbol_items, symbol_index, symbol_labels = [], {}, []
            log.warning("completions.json not found; completion/hover docs limited")
            return

        data = json.loads(assets_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("completions.json is not a list")

        symbol_items = data
        idx: Dict[str, dict] = {}
        for item in data:
            label = str(item.get("label", ""))
            clean = re.sub(r"\(\)$", "", label)
            if clean:
                idx[clean] = item
        symbol_index = idx
        symbol_labels = sorted(symbol_index.keys())
        log.info(f"Loaded {len(symbol_labels)} symbols")
    except Exception as e:
        log.error(f"Failed to load symbols: {e}")
        symbol_items, symbol_index, symbol_labels = [], {}, []

def new_state(text: str = "") -> dict:
    return {
        "text": text,
        "dirty": True,
        "tokens": [],
        "start_positions": [],
        "line_offsets": [],
        "definitions": {},
        "defs_built": False,
        "last_token_time": 0.0,
        "inlay_dirty": True,
        "inlay_hints": [],
    }

def ensure_state(uri: str) -> None:
    if uri not in document_states:
        text = read_document(uri) or ""
        document_states[uri] = new_state(text)

def ensure_tokens(uri: str) -> dict:
    ensure_state(uri)
    st = document_states[uri]
    now = time.time()
    if st["dirty"] and (now - st["last_token_time"]) >= DEBOUNCE_INTERVAL:
        clear(uri)
        toks = tokenize(st["text"], uri, True)
        st["tokens"] = toks
        st["start_positions"] = [t.starting_pos for t in toks]
        st["line_offsets"] = compute_line_offsets(st["text"])
        st["dirty"] = False
        st["last_token_time"] = now
        st["defs_built"] = False
        st["inlay_dirty"] = True
        log.info(f"Tokenized {uri}: {len(toks)} tokens")
    elif not st["tokens"]:
        toks = tokenize(st["text"], uri, True)
        st["tokens"] = toks
        st["start_positions"] = [t.starting_pos for t in toks]
        st["line_offsets"] = compute_line_offsets(st["text"])

    return st

_DEF_PATTERN = re.compile(
    r"\b(?:class|function|process|var|def|fun)\b.*?\b(\w+)\b|"
    r"class\s+(\w+)\s*{"
)
def build_definitions(uri: str) -> None:
    st = document_states.get(uri)
    if not st:
        return
    defs: Dict[str, dict] = {}
    for i, line in enumerate(st["text"].splitlines()):
        clean = line.split("//", 1)[0]
        m = _DEF_PATTERN.search(clean)
        if not m:
            continue
        name = m.group(1) or m.group(2)
        if not name:
            continue
        col = clean.find(name)
        defs[name] = {
            "uri": uri,
            "range": {
                "start": {"line": i, "character": col},
                "end":   {"line": i, "character": col + len(name)},
            },
        }
    st["definitions"] = defs
    st["defs_built"] = True
    log.info(f"Built definitions for {uri}: {len(defs)} entries")

def find_definition_in_state(uri: str, word: str) -> Optional[dict]:
    ensure_state(uri)
    st = document_states[uri]
    if not st["defs_built"]:
        build_definitions(uri)
    return st["definitions"].get(word)

def update_class_index(uri: str, text: str) -> None:
    classes: Dict[str, List[str]] = {}
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
                    methods = re.findall(r'\b(?:function|def|fun|process)\s+(\w+)\s*\(', body)
                    classes[class_name] = methods
                    break
    file_class_index[uri] = {**classes, "_text_hash": [hash(text)]}

def extract_imports(uri: str, content: str) -> List[str]:
    results: List[str] = []
    pattern = re.compile(r'^\s*import\s+["\']([^"\']+)["\']')

    for raw_line in content.splitlines():
        
        line = raw_line.split('//', 1)[0].strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            results.append(m.group(1))

    return results

def resolve_import_uri(from_uri: str, import_path: str) -> Optional[str]:
    base_path = uri_to_path(from_uri)
    if not base_path:
        return None

    imp = Path(import_path)
    
    if imp.is_absolute():
        candidate = imp
    else:
        candidate = (base_path.parent / imp)

    
    if candidate.is_file():
        return path_to_uri(candidate.resolve())

    
    if candidate.suffix == "":
        candidate_jc = candidate.with_suffix('.jc')
        if candidate_jc.is_file():
            return path_to_uri(candidate_jc.resolve())

    return None

def parse_function_signature(line: str) -> Optional[dict]:
    clean = line.split("//", 1)[0]
    m = re.search(
        r"\b(?:function|process|def|fun)\b\s+(\w+)\s*\(\s*([^)]*)\s*\)",
        clean, re.IGNORECASE
    )
    if not m:
        return None

    name = m.group(1)
    raw_params = m.group(2) or ""

    
    params: List[str] = []
    current = []
    depth_round = depth_square = depth_curly = 0

    for ch in raw_params:
        if ch == "(":
            depth_round += 1
        elif ch == ")":
            if depth_round > 0:
                depth_round -= 1
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            if depth_square > 0:
                depth_square -= 1
        elif ch == "{":
            depth_curly += 1
        elif ch == "}":
            if depth_curly > 0:
                depth_curly -= 1

        if ch == "," and depth_round == depth_square == depth_curly == 0:
            param_str = "".join(current).strip()
            if param_str:
                params.append(param_str)
            current = []
        else:
            current.append(ch)

    last_param = "".join(current).strip()
    if last_param:
        params.append(last_param)

    return {"name": name, "params": params}

def get_signature_from_definition(start_uri: str, func_name: str) -> Optional[dict]:
    visited = set()
    def _search(uri: str) -> Optional[dict]:
        if uri in visited:
            return None
        visited.add(uri)
        text = document_states.get(uri, {}).get("text") or read_document(uri) or ""
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
def get_completions(prefix: str) -> List[dict]:
    if not prefix:
        return [it.copy() for it in symbol_items]
    left = bisect.bisect_left(symbol_labels, prefix)
    high = prefix + "\uffff"
    right = bisect.bisect_right(symbol_labels, high)
    return [symbol_index[symbol_labels[i]].copy() for i in range(left, right)]

def _split_top_level_by_comma(tokens: List[Any]) -> List[List[Any]]:
    groups: List[List[Any]] = []
    cur: List[Any] = []
    lvl = 0
    for tt in tokens:
        if tt.type in (Tokens.LPAREN, Tokens.LSPAREN, Tokens.LCPAREN):
            lvl += 1
        elif tt.type in (Tokens.RPAREN, Tokens.RSPAREN, Tokens.RCPAREN):
            lvl -= 1
        if tt.type == Tokens.COMMA and lvl == 0:
            if cur:
                groups.append(cur)
                cur = []
        else:
            cur.append(tt)
    if cur:
        groups.append(cur)
    return groups

def _load_signature_params(uri: str, func_name: str, is_method: bool) -> List[str]:
    item = symbol_index.get(func_name) or (symbol_index.get("." + func_name) if not func_name.startswith(".") else None)
    if item and "signature" in item:
        sig_params = item["signature"].get("parameters", [])
        return [p["label"].split(":")[0].strip() if isinstance(p, dict) else str(p) for p in sig_params]

    if (is_method and "." in func_name) or ((not is_method) and "::" in func_name):
        sep = "." if is_method else "::"
        bare = func_name.split(sep, 1)[1]
        item2 = symbol_index.get(bare) or (symbol_index.get("." + bare) if not bare.startswith(".") else None)
        if item2 and "signature" in item2:
            sig_params = item2["signature"].get("parameters", [])
            return [p["label"].split(":")[0].strip() if isinstance(p, dict) else str(p) for p in sig_params]

    sig_def = get_signature_from_definition(uri, func_name)
    if not sig_def:
        if is_method and "." in func_name:
            sig_def = get_signature_from_definition(uri, func_name.split(".", 1)[1])
        elif (not is_method) and "::" in func_name:
            sig_def = get_signature_from_definition(uri, func_name.split("::", 1)[1])

    return list(sig_def.get("params", [])) if sig_def else []

def handle_initialize(msg: dict) -> None:
    global HIDE_INLAY_HINTS, HIDE_HOVER, HIDE_COMPLETION, HIDE_SIGNATURE_HELP
    rpc_id = msg.get("id")
    params = msg.get("params", {}) or {}
    init_opts = params.get("initializationOptions", {}) or {}

    HIDE_INLAY_HINTS    = bool(init_opts.get("hideInlayHints", False))
    HIDE_HOVER          = bool(init_opts.get("hideHover", False))
    HIDE_COMPLETION     = bool(init_opts.get("hideCompletion", False))
    HIDE_SIGNATURE_HELP = bool(init_opts.get("hideSignatureHelp", False))

    load_symbols()

    capabilities: Dict[str, Any] = {
        "textDocumentSync": {"openClose": True, "change": 1, "save": True},
        "definitionProvider": True,
    }
    if not HIDE_COMPLETION:
        capabilities["completionProvider"] = {
            "resolveProvider": False,
            "triggerCharacters": [".", ":", "<", "_", "="],
        }
    if not HIDE_HOVER:
        capabilities["hoverProvider"] = True
    if not HIDE_SIGNATURE_HELP:
        capabilities["signatureHelpProvider"] = {
            "triggerCharacters": ["(", ","],
            "retriggerCharacters": [")"],
        }
    if not HIDE_INLAY_HINTS:
        capabilities["inlayHintProvider"] = True

    send_message({"id": rpc_id, "result": {"capabilities": capabilities}})

def handle_did_change_configuration(msg: dict) -> None:
    cfg = (
        msg.get("params", {})
           .get("settings", {})
           .get("jmcc-helper", {})
        or {}
    )
    global HIDE_INLAY_HINTS, HIDE_HOVER, HIDE_COMPLETION, HIDE_SIGNATURE_HELP
    HIDE_INLAY_HINTS    = bool(cfg.get("hideInlayHints",    HIDE_INLAY_HINTS))
    HIDE_HOVER          = bool(cfg.get("hideHover",         HIDE_HOVER))
    HIDE_COMPLETION     = bool(cfg.get("hideCompletion",    HIDE_COMPLETION))
    HIDE_SIGNATURE_HELP = bool(cfg.get("hideSignatureHelp", HIDE_SIGNATURE_HELP))
    log.info(f"Config updated: inlay={HIDE_INLAY_HINTS} hover={HIDE_HOVER} completion={HIDE_COMPLETION} signature={HIDE_SIGNATURE_HELP}")

def handle_did_open(msg: dict) -> None:
    try:
        uri = msg["params"]["textDocument"]["uri"]
        text = msg["params"]["textDocument"]["text"].replace("\r", "")
        document_states[uri] = new_state(text)
        clear(uri)
        log.info(f"Opened {uri}")
    except Exception as e:
        log.error(f"didOpen error: {e}")


def handle_did_change(msg: dict) -> None:
    try:
        uri = msg["params"]["textDocument"]["uri"]
        changes = msg["params"].get("contentChanges", [])
        if not changes:
            return
        text = str(changes[0].get("text", "")).replace("\r", "")
        st = document_states.get(uri)
        if st:
            st["text"] = text
            st["dirty"] = True
            st["defs_built"] = False
            st["inlay_dirty"] = True
        else:
            document_states[uri] = new_state(text)
        clear(uri)
        log.info(f"Changed {uri}")
    except Exception as e:
        log.error(f"didChange error: {e}")

from typing import Any, Dict, List, Optional, Set, Tuple
import re

_symbol_completion_cache: Dict[str, Tuple[List[Dict[str, Any]], int]] = {}

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
        lbl = it.get("insertText", it["label"])
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

def _load_document_symbols(uri: str) -> Dict[str, Any]:
    st = ensure_tokens(uri)
    if not st.get("defs_built", False):
        build_definitions(uri)
    return st["definitions"]


def get_class_method_signatures(text: str, class_name: str, method_name: str) -> List[str]:
    sigs, in_cls, depth = [], False, 0
    for line in text.splitlines():
        stripped = line.lstrip()
        if not in_cls:
            if re.match(rf'class\s+{re.escape(class_name)}\b', stripped):
                in_cls = True
                depth = stripped.count("{") - stripped.count("}")
            continue
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            in_cls = False
            continue
        m = re.match(
            rf'\s*(?P<keyword>function|def|fun|process)\s+{re.escape(method_name)}\s*\((?P<params>.*?)\)\s*(?P<return>->\s*.*?)?\s*\{{',
            line
        )
        if m:
            keyword = m.group("keyword")
            params = m.group("params").strip()
            ret = m.group("return").strip() if m.group("return") else ""
            if keyword == "process":
                sig = f"{keyword} {method_name}({params})"
            else:
                sig = f"{keyword} {method_name}({params}){(' ' + ret) if ret else ''}"
            sigs.append(sig)
    return sigs

def get_free_function_signatures(text: str, func_name: str) -> List[str]:
    sigs, depth = [], 0
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if re.match(r'class\b', stripped) or re.match(r'(?:function|def|fun|process)\b', stripped):
            depth += line.count("{")
        depth -= line.count("}")
        if depth != 0:
            continue
        m = re.match(
            rf'\s*(?P<keyword>function|def|fun|process)\s+{re.escape(func_name)}\s*\((?P<params>[^)]*)\)\s*(?P<return>->\s*.*?)?',
            line
        )
        if m:
            keyword = m.group("keyword")
            params = m.group("params").strip()
            ret = m.group("return").strip() if m.group("return") else ""
            sig_line = line
            if not re.search(r'\{[\s]*$', line):
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('{'):
                        sig_line = line + " " + next_line
                        
            if keyword == "process":
                sig = f"{keyword} {func_name}({params})"
            else:
                sig = f"{keyword} {func_name}({params}){(' ' + ret) if ret else ''}"
            sigs.append(sig)
    return sigs

def _find_var_type_before_pos(text: str, obj_name: str, end_pos: int) -> Optional[str]:
    pattern = re.compile(rf'\b{re.escape(obj_name)}\s*=\s*([A-Za-z_]\w*)\s*\(')
    matches = list(pattern.finditer(text, 0, end_pos))
    if matches:
        last_match = matches[-1]
        return last_match.group(1)

    return None

def _class_methods_for(uri: str, class_name: str) -> List[str]:
    methods: set = set()
    local_methods = file_class_index.get(uri, {}).get(class_name, [])
    methods.update(local_methods)
    log.debug(f"Local methods for '{class_name}': {local_methods}")
    prefix = f"{class_name}."
    symbol_methods = []
    for key in symbol_index.keys():
        if key.startswith(prefix) and not key.startswith("."):
            method_part = key.split(".", 1)[1]
            methods.add(method_part)
            symbol_methods.append(method_part)
    log.debug(f"Symbol index methods for '{class_name}': {symbol_methods}")
    result = sorted(m for m in methods)
    log.debug(f"Final methods list for '{class_name}': {result}")
    return result

def _method_signatures_for(text: str, class_name: str, method_name: str) -> List[str]:
    fqn = f"{class_name}.{method_name}"
    itm = symbol_index.get(fqn)
    if itm and "signature" in itm:
        return [itm["signature"]["label"]]
    return get_class_method_signatures(text, class_name, method_name)

from jmcc_extension import *

def handle_hover(msg: dict) -> None:
    rpc_id = msg.get("id")
    if HIDE_HOVER:
        send_message({"id": rpc_id, "result": None})
        return
    try:
        params  = msg["params"]
        uri     = params["textDocument"]["uri"]
        pos     = params["position"]
        st      = ensure_tokens(uri)
        text    = st["text"]
        offsets = st["line_offsets"]
        prev_hash = file_class_index.get(uri, {}).get("_text_hash")
        if isinstance(prev_hash, list):
            prev_hash = prev_hash[0]
        new_hash = hash(text)
        if prev_hash != new_hash:
            update_class_index(uri, text)
            file_class_index.setdefault(uri, {})["_text_hash"] = [new_hash]
        global_pos = line_col_to_pos(pos["line"], pos["character"], offsets)
        idx = find_token_that_have_pos(uri, global_pos)
        
        if idx < 0 or idx >= len(st["tokens"]):
            send_message({"id": rpc_id, "result": None})
            return
        raw = try_find_object(uri, idx)
        if not raw:
            send_message({"id": rpc_id, "result": None})
            return
        log.debug(f"Hover raw object: '{raw}'")
        sep  = "." if "." in raw else ("::" if "::" in raw else None)
        bare = raw.split(sep)[-1] if sep else raw
        local_defs = _load_document_symbols(uri)
        is_local   = bare in local_defs
        keyword    = None
        if is_local:
            df    = local_defs[bare]
            ln    = df["range"]["start"]["line"]
            first = text.splitlines()[ln].lstrip()
            m_kw  = re.match(r'^(class|function|process|fun|def)\b', first)
            if m_kw:
                keyword = m_kw.group(1)
        import_uris: List[str] = []
        for imp in extract_imports(uri, text):
            r = resolve_import_uri(uri, imp)
            if r:
                import_uris.append(r)
        
        contents: Optional[Dict[str, Any]] = None
        
        if sep == ".":
            parts = raw.split(".")
            obj_name = parts[0]
            method_name = parts[-1]
            obj_start_pos = line_and_offset_to_pos(text, pos["line"], pos["character"] - len(raw))
            var_type = _find_var_type_before_pos(text, obj_name, obj_start_pos)
            log.debug(f"Hover on method call: raw='{raw}', obj_name='{obj_name}', method_name='{method_name}', var_type='{var_type}'")
            if var_type:
                class_methods = _class_methods_for(uri, var_type)
                method_belongs = method_name in class_methods
                log.debug(f"Object type '{var_type}' found. Methods: {class_methods}. Method belongs: {method_belongs}")
                if method_belongs:
                    sigs = _method_signatures_for(text, var_type, method_name)
                    if sigs:
                        body = "\n".join(sigs)
                        contents = {
                            "kind": "markdown",
                            "value": f"```justcode\nclass {var_type}\n{body}\n```"
                        }
                        log.debug(f"Found specific method signatures for {var_type}.{method_name}")
                    else:
                        fqn = f"{var_type}.{method_name}"
                        itm = symbol_index.get(fqn)
                        if itm and "documentation" in itm:
                            doc = itm["documentation"]
                            contents = doc if isinstance(doc, dict) else {"kind": "markdown", "value": str(doc)}
                            log.debug(f"Found documentation for {fqn} in symbol_index")
            if not var_type: 
                log.debug(f"Object type for '{obj_name}' not found. Trying generic method '.{method_name}'")
                itm = symbol_index.get("." + method_name)
                if itm:
                    if "documentation" in itm:
                        doc = itm["documentation"]
                        contents = doc if isinstance(doc, dict) else {"kind": "markdown", "value": str(doc)}
                        log.debug(f"Found documentation for generic method '.{method_name}'")
                    elif "signature" in itm:
                        contents = {"kind": "markdown", "value": f"```justcode\n{itm['signature']['label']}\n```"}
                        log.debug(f"Found signature for generic method '.{method_name}'")

        elif sep == "::":
            itm = symbol_index.get(raw) or symbol_index.get(bare)  
            if itm:
                if "documentation" in itm:
                    doc = itm["documentation"]
                    contents = doc if isinstance(doc, dict) else {"kind":"markdown","value":str(doc)}
                elif "signature" in itm:
                    contents = {"kind":"markdown","value":f"```justcode\n{itm['signature']['label']}\n```"}
        
        if contents is None and sep is None:
            var_class = _find_var_type_before_pos(text, bare, global_pos)
            if var_class:
                lines = [f"class {var_class}"]
                for mth in _class_methods_for(uri, var_class):
                    sigs = _method_signatures_for(text, var_class, mth)
                    if sigs:
                        for s in sigs:
                            lines.append(f"  {s}")
                    else:
                        lines.append(f"  {mth}")
                contents = {
                    "kind": "markdown",
                    "value": "```justcode\n" + "\n".join(lines) + "\n```"
                }
        
        if contents is None and is_local and keyword == "class":
            lines = [f"class {bare}"]
            for mth in _class_methods_for(uri, bare):
                sigs = _method_signatures_for(text, bare, mth)
                if sigs:
                    for s in sigs:
                        lines.append(f"  {s}")
                else:
                    lines.append(f"  {mth}")
            contents = {
                "kind": "markdown",
                "value": "```justcode\n" + "\n".join(lines) + "\n```"
            }
        
        if contents is None and is_local and keyword in ("function", "process", "fun", "def"):
            itm = symbol_index.get(bare)
            if itm and "signature" in itm:
                contents = {"kind":"markdown","value":f"```justcode\n{itm['signature']['label']}\n```"}
            else:
                sigs = get_free_function_signatures(text, bare)
                if sigs:
                    contents = {"kind":"markdown","value":"```justcode\n" + "\n".join(sigs) + "\n```"}
                else:
                    def_line = text.splitlines()[df["range"]["start"]["line"]]
                    param_pattern = re.compile(rf'\b{keyword}\s+{re.escape(bare)}\s*\(([^)]*)\)')
                    param_match = param_pattern.search(def_line)
                    class_name = None
                    for cname, methods in file_class_index.get(uri, {}).items():
                        if cname != "_text_hash" and bare in methods:
                            class_name = cname
                            break
                    
                    if param_match:
                        params = param_match.group(1)
                        if class_name:
                            contents = {"kind":"markdown","value":f"```justcode\nclass {class_name}\n{keyword} {bare}({params})\n```"}
                        else:
                            contents = {"kind":"markdown","value":f"```justcode\n{keyword} {bare}({params})\n```"}
                    else:
                        if class_name:
                            contents = {"kind":"markdown","value":f"```justcode\nclass {class_name}\n{keyword} {bare}(...)\n```"}
                        else:
                            contents = {"kind":"markdown","value":f"```justcode\n{keyword} {bare}(...)\n```"}
        if contents is None:
            itm = symbol_index.get(bare) or symbol_index.get("." + bare)
            if itm:
                if "documentation" in itm:
                    doc = itm["documentation"]
                    contents = doc if isinstance(doc, dict) else {"kind":"markdown","value":str(doc)}
                elif "signature" in itm:
                    contents = {"kind":"markdown","value":f"```justcode\n{itm['signature']['label']}\n```"}
        
        if contents is None:
            for u in [uri] + import_uris:
                sd = get_signature_from_definition(u, bare)
                if sd:
                    params = ", ".join(sd["params"])
                    contents = {"kind":"markdown","value":f"```justcode\n{bare}({params})\n```"}
                    break
        
        if contents is None:
            send_message({"id": rpc_id, "result": None})
            return
        start_line, start_ch = pos_to_line_col(st["tokens"][idx].starting_pos, offsets)
        end_idx = idx
        temp_idx = idx
        while temp_idx + 1 < len(st["tokens"]):
            next_tok = st["tokens"][temp_idx + 1]
            if next_tok.type in (Tokens.DOT, Tokens.DOUBLE_COLON, Tokens.VARIABLE):
                end_idx = temp_idx + 1
                temp_idx += 1
            else:
                break
        last = st["tokens"][end_idx]
        end_line, end_ch = pos_to_line_col(last.ending_pos + 1, offsets)
        
        send_message({
            "id": rpc_id,
            "result": {
                "contents": contents,
                "range": {
                    "start": {"line": start_line, "character": start_ch},
                    "end":   {"line": end_line,   "character": end_ch},
                }
            }
        })
    except Exception as e:
        log.error(f"hover error: {e}", exc_info=True)
        send_message({"id": rpc_id, "result": None})

def handle_definition(msg: dict) -> None:
    
    rpc_id = msg.get("id")
    try:
        uri = msg["params"]["textDocument"]["uri"]
        pos = msg["params"]["position"]
        st = ensure_tokens(uri)
        offsets = st["line_offsets"]
        global_pos = line_col_to_pos(pos["line"], pos["character"], offsets)
        idx = bisect.bisect_right(st["start_positions"], global_pos) - 1
        if idx < 0 or idx >= len(st["tokens"]):
            send_message({"id": rpc_id, "result": None})
            return
        word = try_find_object(uri, idx)
        if not word:
            send_message({"id": rpc_id, "result": None})
            return
        loc = find_definition_in_state(uri, word)
        if loc:
            send_message({"id": rpc_id, "result": loc})
            return
        for imp in extract_imports(uri, st["text"]):
            iuri = resolve_import_uri(uri, imp)
            if not iuri:
                continue
            ensure_state(iuri)
            loc = find_definition_in_state(iuri, word)
            if loc:
                send_message({"id": rpc_id, "result": loc})
                return
        send_message({"id": rpc_id, "result": None})
    except Exception as e:
        log.error(f"definition error: {e}")
        send_message({"id": rpc_id, "result": None})

def handle_signature_help(msg: dict) -> None:
    rpc_id = msg.get("id")
    if HIDE_SIGNATURE_HELP:
        send_message({"id": rpc_id, "result": None})
        return
    try:
        uri = msg["params"]["textDocument"]["uri"]
        pos = msg["params"]["position"]
        st = document_states.get(uri)
        if not st:
            send_message({"id": rpc_id, "result": None})
            return
        lines = st["text"].splitlines()
        if pos["line"] < 0 or pos["line"] >= len(lines):
            send_message({"id": rpc_id, "result": None})
            return
        line = lines[pos["line"]]
        char = pos["character"]
        bracket = -1
        depth = 0
        for i in range(char - 1, -1, -1):
            c = line[i]
            if c == ")":
                depth += 1
            elif c == "(":
                if depth == 0:
                    bracket = i
                    break
                depth -= 1
        if bracket < 0:
            send_message({"id": rpc_id, "result": None})
            return
        start = bracket - 1
        while start >= 0 and (line[start].isalnum() or line[start] in "_:<."):
            start -= 1
        func = line[start + 1: bracket].strip()
        if not func:
            send_message({"id": rpc_id, "result": None})
            return
        sig_data = None
        item = symbol_index.get(func)
        if item and "signature" in item:
            sig_data = item["signature"]
        else:
            sig_def = get_signature_from_definition(uri, func)
            if sig_def:
                full_params_str = ", ".join(sig_def["params"])
                sig_data = {
                    "label": f"{func}({full_params_str})",
                    "parameters": [{"label": p} for p in sig_def["params"]],
                    "documentation": ""
                }
        if not sig_data:
            send_message({"id": rpc_id, "result": None})
            return
        inner = line[bracket + 1: char]
        depth_r = depth_s = depth_c = 0
        active_param_index = 0
        for ch in inner:
            if ch == "(":
                depth_r += 1
            elif ch == ")":
                if depth_r > 0:
                    depth_r -= 1
            elif ch == "[":
                depth_s += 1
            elif ch == "]":
                if depth_s > 0:
                    depth_s -= 1
            elif ch == "{":
                depth_c += 1
            elif ch == "}":
                if depth_c > 0:
                    depth_c -= 1
            elif ch == "," and depth_r == depth_s == depth_c == 0:
                active_param_index += 1
        result = {
            "signatures": [
                {
                    "label": sig_data.get("label", f"{func}(...)"),
                    "documentation": sig_data.get("documentation", ""),
                    "parameters": [
                        {"label": (p.get("label") if isinstance(p, dict) else str(p)),
                         "documentation": (p.get("documentation", "") if isinstance(p, dict) else "")}
                        for p in sig_data.get("parameters", [])
                    ]
                }
            ],
            "activeSignature": 0,
            "activeParameter": active_param_index
        }
        send_message({"id": rpc_id, "result": result})
    except Exception as e:
        log.error(f"signatureHelp error: {e}")
        send_message({"id": rpc_id, "result": None})
def handle_inlay_hints(msg: dict):
    rpc_id = msg.get("id")
    uri = msg["params"]["textDocument"]["uri"]

    if HIDE_INLAY_HINTS:
        send_message({"id": rpc_id, "result": []})
        return

    st = ensure_tokens(uri)
    if not st["inlay_dirty"]:
        hints = []
        for h in st["inlay_hints"]:
            label = h["label"]
            if isinstance(label, str):
                label = label.split(":", 1)[0].strip() + ":"
            hints.append({**h, "label": label})
        send_message({"id": rpc_id, "result": hints})
        return

    tokens = st["tokens"]
    offsets = st["line_offsets"]
    hints = []
    idx = 0

    def is_definition_header(pos: int) -> bool:
        j = pos - 1
        while j >= 0 and tokens[j].type in (Tokens.NEXT_LINE, Tokens.SEMICOLON, Tokens.COLON):
            j -= 1
        return j >= 0 and tokens[j].type in (
            Tokens.FUNCTION_DEFINE,
            Tokens.PROCESS_DEFINE,
            Tokens.CLASS_DEFINE,
        )

    while idx < len(tokens):
        t = tokens[idx]
        func_name = None
        open_paren = None
        static_call = False
        method_call = False

        
        if t.type == Tokens.VARIABLE and not is_definition_header(idx) and idx + 1 < len(tokens):
            if tokens[idx + 1].type == Tokens.LPAREN:
                func_name = t.value
                open_paren = idx + 1
            elif tokens[idx + 1].type == Tokens.SUBSTRING and idx + 2 < len(tokens) and tokens[idx + 2].type == Tokens.LPAREN:
                func_name = t.value
                open_paren = idx + 2

        
        if func_name is None and t.type == Tokens.VARIABLE and idx + 2 < len(tokens):
            if (
                tokens[idx + 1].type == Tokens.DOUBLE_COLON
                and tokens[idx + 2].type == Tokens.VARIABLE
                and not is_definition_header(idx)
            ):
                name = f"{t.value}::{tokens[idx + 2].value}"
                pos2 = idx + 3
                if pos2 < len(tokens) and tokens[pos2].type == Tokens.LPAREN:
                    func_name = name
                    open_paren = pos2
                    static_call = True
                elif (
                    pos2 < len(tokens)
                    and tokens[pos2].type == Tokens.SUBSTRING
                    and pos2 + 1 < len(tokens)
                    and tokens[pos2 + 1].type == Tokens.LPAREN
                ):
                    func_name = name
                    open_paren = pos2 + 1
                    static_call = True

        
        if func_name is None and t.type == Tokens.VARIABLE:
            j = idx
            parts = [t.value]
            while j + 2 < len(tokens) and tokens[j + 1].type == Tokens.DOT and tokens[j + 2].type == Tokens.VARIABLE:
                parts.append(tokens[j + 2].value)
                j += 2
            if len(parts) > 1 and not is_definition_header(idx):
                name = ".".join(parts)
                pos2 = j + 1
                if pos2 < len(tokens) and tokens[pos2].type == Tokens.LPAREN:
                    func_name = name
                    open_paren = pos2
                    method_call = True
                elif (
                    pos2 < len(tokens)
                    and tokens[pos2].type == Tokens.SUBSTRING
                    and pos2 + 1 < len(tokens)
                    and tokens[pos2 + 1].type == Tokens.LPAREN
                ):
                    func_name = name
                    open_paren = pos2 + 1
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

        sig_params = _load_signature_params(uri, func_name, is_method=method_call)
        if not sig_params:
            idx = close_paren + 1
            continue

        
        if any(p.strip().startswith("*") for p in sig_params):
            idx = close_paren + 1
            continue

        args = tokens[open_paren + 1 : close_paren]
        if not args:
            idx = close_paren + 1
            continue

        
        offset = 0
        if func_name and func_name.startswith("repeat::"):
            k = close_paren + 1
            while k < len(tokens) and tokens[k].type in (Tokens.NEXT_LINE, Tokens.SEMICOLON):
                k += 1
            if k < len(tokens) and tokens[k].type == Tokens.LCPAREN:
                bd = 1
                block_end = None
                for m in range(k + 1, len(tokens)):
                    if tokens[m].type == Tokens.LCPAREN:
                        bd += 1
                    elif tokens[m].type == Tokens.RCPAREN:
                        bd -= 1
                        if bd == 0:
                            block_end = m
                            break
                if block_end is not None:
                    vc = 0
                    found = False
                    for tt in tokens[k + 1 : block_end]:
                        if tt.type == Tokens.CYCLE_THING:
                            found = True
                            break
                        if tt.type == Tokens.VARIABLE:
                            vc += 1
                    if found:
                        offset = vc

        
        elif static_call or method_call:
            assign_idx = None
            return_found = False
            for m in range(idx - 1, -1, -1):
                tp3 = tokens[m].type
                if tp3 in (Tokens.SEMICOLON, Tokens.NEXT_LINE):
                    break
                if tp3 == Tokens.ASSIGN:
                    assign_idx = m
                    break
                if tp3 == Tokens.RETURN:
                    return_found = True
                    break
            if assign_idx is not None:
                vc = 0
                p2 = assign_idx - 1
                while p2 >= 0 and tokens[p2].type in (Tokens.VARIABLE, Tokens.COMMA, Tokens.DOT):
                    if tokens[p2].type == Tokens.VARIABLE:
                        vc += 1
                    p2 -= 1
                offset = vc
            elif return_found and len(sig_params) > 1:
                offset = 1

        
        groups = _split_top_level_by_comma(args)
        named = set()
        for grp in groups:
            for i2, tok2 in enumerate(grp):
                if tok2.type == Tokens.ASSIGN and i2 > 0 and grp[i2 - 1].type == Tokens.VARIABLE:
                    named.add(grp[i2 - 1].value)

        active = [p for p in sig_params[offset:] if p not in named]
        if active:
            pidx = 0
            for grp in groups:
                if any(tt.type == Tokens.ASSIGN for tt in grp):
                    continue
                if pidx >= len(active):
                    break
                first_tok = next((x for x in grp if x.type != Tokens.NEXT_LINE), None)
                if first_tok:
                    ln, ch = pos_to_line_col(first_tok.starting_pos, offsets)
                    label = active[pidx].split(":", 1)[0].strip() + ":"
                    hints.append({
                        "position": {"line": ln, "character": ch},
                        "label": label,
                        "kind": 1,
                        "paddingLeft": False,
                        "paddingRight": True,
                    })
                    pidx += 1

        idx = close_paren + 1

    st["inlay_hints"] = hints
    st["inlay_dirty"] = False
    send_message({"id": rpc_id, "result": hints})

def main() -> None:
    log.info("JMCC LSP Server started")
    handlers: Dict[str, Callable[[dict], None]] = {
        "initialize": handle_initialize,
        "workspace/didChangeConfiguration": handle_did_change_configuration,
        "textDocument/didOpen": handle_did_open,
        "textDocument/didChange": handle_did_change,
        "textDocument/completion": handle_completion,
        "textDocument/hover": handle_hover,
        "textDocument/definition": handle_definition,
        "textDocument/signatureHelp": handle_signature_help,
        "textDocument/inlayHint": handle_inlay_hints,
    }

    while True:
        msg = read_message()
        if msg is None:
            log.info("Input stream closed; exiting server")
            break
        if not isinstance(msg, dict) or not msg:
            continue

        method = msg.get("method")
        if method == "exit":
            log.info("Exit requested")
            break

        handler = handlers.get(method)
        if handler:
            handler(msg)
        else:
            if method not in ("textDocument/didSave",):
                log.warning(f"Unhandled method: {method}")

if __name__ == "__main__":
    main()