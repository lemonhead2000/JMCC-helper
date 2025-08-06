import sys
import json
import os
import re
import logging
from urllib.parse import urlparse, urlunparse, urljoin, unquote
from urllib.request import urlopen

logging.basicConfig(
    level=logging.INFO,
    format="[LSP] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)

def log(msg):
    print(f"[LSP] {msg}", file=sys.stderr)
    sys.stderr.flush()
try:
    import jmcc_extension
except Exception as e:
    log(f"Unhandled error: {e}")
def load_json(filename):
    try:
        server_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(server_dir, "assets", filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            log(f"Loaded {len(data)} entries from {filename}")
            return data
    except Exception as e:
        log(f"Failed to load {filename}: {e}")
        return {} if "hover" in filename else []

completions_db = load_json("completions.json")
hover_db = load_json("hover.json")

documents = {}

def read_message():
    try:
        headers = {}
        while True:
            line = sys.stdin.buffer.readline().decode("utf-8", errors="replace").strip()
            if line == "":
                break
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()

        if "Content-Length" not in headers:
            log("No Content-Length header")
            return None

        content_length = int(headers["Content-Length"])
        body = sys.stdin.buffer.read(content_length).decode("utf-8")
        return json.loads(body)
    except Exception as e:
        log(f"Error reading message: {e}")
        return None

def send_message(msg):
    try:
        body = json.dumps(msg, ensure_ascii=False)
        body_bytes = body.encode("utf-8")
        headers = (
            f"Content-Length: {len(body_bytes)}\r\n"
            f"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n"
            f"\r\n"
        )
        headers_bytes = headers.encode("utf-8")
        sys.stdout.buffer.write(headers_bytes + body_bytes)
        sys.stdout.buffer.flush()
    except Exception as e:
        log(f"Error sending message: {e}")


def uri_to_path(uri):
    try:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            return None
        path = unquote(parsed.path)
        if os.name == "nt":
            if path.startswith("/"):
                path = path[1:]
            path = path.replace("/", "\\")
        return path
    except Exception as e:
        log(f"Error parsing URI {uri}: {e}")
        return None
def path_to_uri(path):
    path = os.path.abspath(path)
    if os.name == "nt":
        path = path.replace("\\", "/")
    return f"file:///{path}"

def read_document(uri):
    path = uri_to_path(uri)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        log(f"Failed to read {path}: {e}")
        return None

def get_word_at_position(lines, line_num, char):
    if line_num >= len(lines):
        return None, -1, -1
    line = lines[line_num]
    if not line or char > len(line):
        return None, -1, -1

    start = char
    while start > 0 and (line[start-1].isalnum() or line[start-1] in [":", "_"]):
        start -= 1
    end = char
    while end < len(line) and (line[end].isalnum() or line[end] in [":", "_"]):
        end += 1

    word = line[start:end]
    return word, start, end

def main():
    log("Server starting")

    while True:
        try:
            msg = read_message()
            if msg is None:
                continue

            method = msg.get("method")
            rpc_id = msg.get("id")

            if method == "initialize":
                log("initialize")
                send_message({
        "id": rpc_id,
        "result": {
            "capabilities": {
                "textDocumentSync": {
                    "openClose": True,
                    "change": 1,
                    "save": True
                },
                "completionProvider": {
                    "resolveProvider": False,
                    "triggerCharacters": [":", "="]
                },
                "hoverProvider": True,
                "definitionProvider": True,
                "signatureHelpProvider": {
                    "triggerCharacters": ["(", ","],
                    "retriggerCharacters": [")"]
                },
                "inlayHintProvider": True
            }
        }
    })

            elif method == "initialized":
                log("Client initialized")

            elif method == "textDocument/didOpen":
                uri = msg["params"]["textDocument"]["uri"]
                text = msg["params"]["textDocument"]["text"]
                documents[uri] = text.replace("\r","")
                jmcc_extension.clear(uri)
                jmcc_extension.tokenize(documents[uri], uri, True)

            elif method == "textDocument/didChange":
                uri = msg["params"]["textDocument"]["uri"]
                content = msg["params"]["contentChanges"][0]["text"]
                documents[uri] = content.replace("\r","")
                jmcc_extension.clear(uri)
                jmcc_extension.tokenize(documents[uri], uri, True)

            elif method == "textDocument/completion":
                uri = msg["params"]["textDocument"]["uri"]
                pos = msg["params"]["position"]
                line_num = pos["line"]
                char = pos["character"]

                if uri not in documents:
                    send_message({"id": rpc_id, "result": None})
                    continue

                lines = documents[uri].splitlines()
                if line_num >= len(lines):
                    send_message({"id": rpc_id, "result": None})
                    continue

                line = lines[line_num]
                text_before = line[:char]
                match = re.search(r"[\w:]*$", text_before)
                if not match:
                    prefix = ""
                else:
                    prefix = match.group(0)
                prefix = prefix.strip()

                filtered_items = []
                for item in completions_db:
                    label = item.get("label", "")
                    if prefix and not label.startswith(prefix):
                        continue
                    start_char = char - len(prefix)
                    if start_char < 0:
                        start_char = 0
                    text_edit = {
                        "range": {
                            "start": {"line": line_num, "character": start_char},
                            "end": {"line": line_num, "character": char}
                        },
                        "newText": label 
                    }

                    new_item = {**item}
                    if "insertText" in new_item:
                        del new_item["insertText"]
                    new_item["textEdit"] = text_edit

                    filtered_items.append(new_item)

                send_message({
                    "id": rpc_id,
                    "result": {
                        "isIncomplete": False,
                        "items": filtered_items
                    }
                })

            elif method == "textDocument/hover":
                log("Hover requested")
                uri = msg["params"]["textDocument"]["uri"]
                pos = msg["params"]["position"]
                line_num = pos["line"]
                char = pos["character"]

                if uri not in documents:
                    log("Document not found")
                    send_message({"id": rpc_id, "result": None})
                    continue

                if line_num >= documents[uri].count("\n")+1:
                    send_message({"id": rpc_id, "result": None})
                    continue
                
                token_index = jmcc_extension.find_token_that_have_pos(uri, jmcc_extension.line_and_offset_to_pos(documents[uri], line_num, char))
                start, end = jmcc_extension.pos_to_line_and_offset(documents[uri], jmcc_extension.global_tokens[uri][token_index].starting_pos, jmcc_extension.global_tokens[uri][token_index].ending_pos)
                word= jmcc_extension.try_find_object(uri, token_index)
                if not word:
                    send_message({"id": rpc_id, "result": None})
                    continue

                log(f"Hover word detected: '{word}'")
                
                if word in hover_db:
                    log(f"Hover found in JSON for '{word}'")
                    send_message({
                        "id": rpc_id,
                        "result": {
                            "contents": {
                                "kind": "markdown",
                                "value": hover_db[word]
                            },
                            "range": {
                                "start": {"line": start[0], "character": start[1]},
                                "end": {"line": end[0], "character": end[1]}
                            }
                        }
                    })
                else:
                    log(f"No hover found for '{word}' in hover.json")
                    send_message({"id": rpc_id, "result": None})

            elif method == "textDocument/definition":
                uri = msg["params"]["textDocument"]["uri"]
                pos = msg["params"]["position"]
                line_num = pos["line"]
                char = pos["character"]

                if uri not in documents:
                    log("Document not found")
                    send_message({"id": rpc_id, "result": None})
                    continue

                lines = documents[uri].splitlines()
                word, start_char, _ = get_word_at_position(lines, line_num, char)
                if not word:
                    send_message({"id": rpc_id, "result": None})
                    continue

                result = find_definition_in_file(uri, word)
                if result:
                    send_message({"id": rpc_id, "result": result})
                    continue

                imports = extract_imports(uri, documents[uri])
                for import_path in imports:
                    imported_uri = resolve_import_uri(uri, import_path)
                    if not imported_uri:
                        continue
                    result = find_definition_in_file(imported_uri, word)
                    if result:
                        send_message({"id": rpc_id, "result": result})
                        break
                else:
                    send_message({"id": rpc_id, "result": None})
            
            elif method == "textDocument/signatureHelp":
                log("Signature help requested")
                uri = msg["params"]["textDocument"]["uri"]
                pos = msg["params"]["position"]
                line_num = pos["line"]

                if uri not in documents:
                    send_message({"id": rpc_id, "result": None})
                    continue

                lines = documents[uri].splitlines()
                if line_num >= len(lines):
                    send_message({"id": rpc_id, "result": None})
                    continue

                line = lines[line_num]
                char = pos["character"]

                bracket_pos = -1
                paren_count = 0
                for i in range(char - 1, -1, -1):
                    c = line[i]
                    if c == ")":
                        paren_count += 1
                    elif c == "(":
                        if paren_count == 0:
                            bracket_pos = i
                            break
                        else:
                            paren_count -= 1
                    elif c in " \t\n;{}":
                        if paren_count == 0:
                            break

                if bracket_pos == -1:
                    send_message({"id": rpc_id, "result": None})
                    continue

                func_start = bracket_pos - 1
                while func_start > 0 and (line[func_start - 1].isalnum() or line[func_start - 1] in "_"):
                    func_start -= 1
                func_name = line[func_start:bracket_pos].strip()

                if not func_name:
                    send_message({"id": rpc_id, "result": None})
                    continue

                log(f"Resolving signature for: '{func_name}'")

                signatures = load_signatures()
                sig_data = signatures.get(func_name)

                if sig_data:
                    log(f"Signature found in JSON for '{func_name}'")
                else:

                    log(f"🔍 Signature not in JSON, searching in code...")
                    sig_data = get_signature_from_definition(uri, func_name)
                    if sig_data:
                        log(f"Signature generated from code: {sig_data['label']}")
                    else:
                        log(f"No signature found for '{func_name}' in JSON or code")
                        send_message({"id": rpc_id, "result": None})
                        continue

                inner = line[bracket_pos + 1:char]
                arg_idx = inner.count(",")

                params = []
                if isinstance(sig_data.get("params"), list):
                    params = [{"label": p} for p in sig_data["params"]]
                else:
                    params = []

                send_message({
                    "id": rpc_id,
                    "result": {
                        "signatures": [
                            {
                                "label": sig_data.get("label", f"{func_name}(...)"),
                                "documentation": sig_data.get("doc", ""),
                                "parameters": params
                            }
                        ],
                        "activeSignature": 0,
                        "activeParameter": arg_idx
                    }
                })
 
            elif method == "textDocument/inlayHint":
                log("Inlay hint requested")
                uri = msg["params"]["textDocument"]["uri"]
                if uri not in documents:
                    send_message({"id": rpc_id, "result": []})
                    continue
                lines = documents[uri].splitlines()
                hints = []
                definition_lines = set()
                for line_num, line in enumerate(lines):
                    stripped = line.strip()
                    if re.match(r"\b(?:function|process|inline\s+function)\b", stripped):
                        definition_lines.add(line_num)

                for line_num, line in enumerate(lines):
                    if line_num in definition_lines:
                        continue
                    for match in re.finditer(r"\b(\w+)\s*\(([^)]*)\)", line):
                        func_name = match.group(1)
                        args_content = match.group(2)
                        if not args_content.strip():
                            continue
                        args = []
                        arg_positions = []
                        paren_start = match.start(2)
                        content = args_content
                        arg_start_rel = 0
                        i = 0
                        while i < len(content):
                            if content[i] in " \t":
                                i += 1
                                continue
                            start = i
                            while i < len(content) and content[i] not in ",)":
                                i += 1
                            raw_arg = content[start:i].strip()
                            if raw_arg:
                                arg_start_abs = paren_start + start
                                arg_end_abs = paren_start + i
                                args.append(raw_arg)
                                arg_positions.append((arg_start_abs, arg_end_abs, raw_arg))
                            if i < len(content) and content[i] == ",":
                                i += 1
                            else:
                                break
                        sig = get_signature_from_definition(uri, func_name)
                        if not sig or not isinstance(sig.get("params"), list):
                            continue

                        params = sig["params"]
                        for idx, (start_abs, end_abs, raw_arg) in enumerate(arg_positions):
                            if "=" in raw_arg and raw_arg.split("=")[0].strip().isidentifier():
                                continue
                            if idx >= len(params):
                                continue
                            param_name = params[idx]
                            if not param_name:
                                continue
                            char_pos = find_argument_position(line, raw_arg, match.start(1))
                            if char_pos == -1:
                                continue

                            hints.append({
                                "position": {
                                    "line": line_num,
                                    "character": char_pos
                                },
                                "label": f"{param_name}:",
                                "kind": 1,
                                "paddingLeft": False,
                                "paddingRight": True
                            })

                log(f"Inlay hints sent: {len(hints)}")
                send_message({
                    "id": rpc_id,
                    "result": hints
                })

            elif method == "exit":
                log("Exit")
                break

            else:
                if method not in ["textDocument/didSave", "textDocument/didChange"]:
                    log(f"Unknown method: {method}")

        except Exception as e:
            log(f"Unhandled error: {e}")

    log("Server stopped")
    sys.exit(0)

def extract_imports(uri, content):
    imports = []
    for i, line in enumerate(content.splitlines()):
        line = line.strip()
        if line.startswith("import ") and '"' in line:
            try:
                path = line.split('"')[1]
                imports.append(path)
            except Exception as e:
                log(f"Failed to parse import line: {line}")
                continue
    return imports


def resolve_import_uri(from_uri, import_path):
    from_path = uri_to_path(from_uri)
    if not from_path:
        log(f"Can't resolve path from URI: {from_uri}")
        return None
    dir_path = os.path.dirname(from_path)
    resolved_path = os.path.normpath(os.path.join(dir_path, import_path))
    if not os.path.exists(resolved_path):
        log(f"Imported file does not exist: {resolved_path}")
        return None
    if not os.path.isfile(resolved_path):
        log(f"Imported path is not a file: {resolved_path}")
        return None

    uri = path_to_uri(resolved_path)
    return uri

def find_definition_in_file(uri, word):
    content = documents.get(uri)
    if not content:
        content = read_document(uri)
        if not content:
            log(f"Failed to read content of {uri}")
            return None
    else:
        log(f"Using cached document for {uri}")

    lines = content.splitlines()

    pattern = re.compile(
        rf"\b(?:def|class|function|process|var)\b.*?\b{re.escape(word)}\b|"
        rf"class\s+{re.escape(word)}\s*{{"
    )

    for i, line in enumerate(lines):
        clean_line = line.split("//")[0].strip()

        if pattern.search(clean_line):
            idx = line.find(word)
            if idx != -1:
                return {
                    "uri": uri,
                    "range": {
                        "start": {"line": i, "character": idx},
                        "end": {"line": i, "character": idx + len(word)}
                    }
                }

    log(f"No definition found for '{word}' in {uri}")
    return None

signatures_db = None
def load_signatures():
    global signatures_db
    if signatures_db is not None:
        return signatures_db
    try:
        server_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(server_dir, "assets", "signatures.json")
        with open(path, "r", encoding="utf-8") as f:
            signatures_db = json.load(f)
            log(f"Loaded {len(signatures_db)} signatures")
            return signatures_db
    except Exception as e:
        log(f"Failed to load signatures: {e}")
        return {}

def parse_function_signature(line):
    """
    Парсит строки вида:
        function __init__(self: vector2d, x: number, y: number) -> vector2d
        inline function __multiply__(primal: text, secondary: number) -> text
        function get_length(self: vector2d)
    """
    line = line.split("//")[0].strip()
    is_inline = "inline" in line
    clean_line = re.sub(r"\binline\b", "", line).strip()
    match = re.search(
        r"\b(?:function|process)\b\s+(\w+)\s*\(\s*([^)]*)\s*\)\s*(?:->\s*(\w+))?",
        clean_line,
        re.IGNORECASE
    )
    if not match:
        return None

    func_name = match.group(1)
    params_str = match.group(2) or ""
    return_type = match.group(3)
    if params_str.strip():
        raw_params = [p.strip() for p in params_str.split(",")]
        param_names = []
        for p in raw_params:
            param_name = re.split(r"[:=]", p)[0].strip()
            if param_name:
                param_names.append(param_name)
    else:
        param_names = []
    params_part = ", ".join(param_names)
    return_part = f" -> {return_type}" if return_type else ""
    label = f"{func_name}({params_part}){return_part}"

    return {
        "name": func_name,
        "params": param_names,
        "label": label,
        "return_type": return_type,
        "is_inline": is_inline
    }

def get_signature_from_definition(start_uri, func_name):
    visited_uris = set()

    def search(uri):
        if uri in visited_uris:
            log(f"Already visited: {uri}")
            return None
        visited_uris.add(uri)

        content = documents.get(uri) or read_document(uri)
        if not content:
            log(f"Failed to read {uri}")
            return None

        for line in content.splitlines():
            sig = parse_function_signature(line)
            if sig and sig["name"] == func_name:
                return sig

        imports = extract_imports(uri, content)
        for imp_path in imports:
            imported_uri = resolve_import_uri(uri, imp_path)
            if not imported_uri:
                log(f"Failed to resolve import: {imp_path}")
                continue
            log(f"Import found: {imp_path} → {imported_uri}")
            result = search(imported_uri)
            if result:
                return result

        return None

    return search(start_uri)

def find_argument_position(line, arg, start_pos):
    pattern = r'\b' + re.escape(arg) + r'\b'
    match = re.search(pattern, line[start_pos:])
    if match:
        return start_pos + match.start()
    return -1
    
if __name__ == "__main__":
    main()