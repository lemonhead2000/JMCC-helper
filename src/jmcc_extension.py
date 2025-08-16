global_text = {}
global_tokens = {}
global_new = {}


def new(source):
    global global_new
    global_new[source] = global_new.setdefault(source, -1) + 1
    return global_new[source]


class Tokens:
    __slots__ = ()
    NONE: int = -1
    NUMBER: int = 0
    PLUS_NUMBER: int = 1
    MINUS_NUMBER: int = 2
    STRING: int = 3
    SUBSTRING: int = 4
    LPAREN: int = 5
    RPAREN: int = 6
    LSPAREN: int = 7
    RSPAREN: int = 8
    LCPAREN: int = 9
    RCPAREN: int = 10
    VARIABLE: int = 11
    COMMA: int = 12
    DOT: int = 13
    COLON: int = 14
    SEMICOLON: int = 15
    ASSIGN: int = 16
    PLUS: int = 17
    MINUS: int = 18
    MULTIPLY: int = 19
    DIVIDE: int = 20
    DEG: int = 21
    PR: int = 22
    INLINE_VARIABLE: int = 23
    LOCAL_VARIABLE: int = 24
    GAME_VARIABLE: int = 25
    SAVE_VARIABLE: int = 26
    BRACKET_VARIABLE: int = 27
    PLAIN_STRING: int = 28
    LEGACY_STRING: int = 29
    MINIMESSAGE_STRING: int = 30
    JSON_STRING: int = 31
    SNBT: int = 32
    IF: int = 33
    ELSE: int = 34
    FUNCTION_DEFINE: int = 35
    PROCESS_DEFINE: int = 36
    VAR_DEFINE: int = 37
    INLINE_DEFINE: int = 38
    LOCAL_DEFINE: int = 39
    GAME_DEFINE: int = 40
    SAVE_DEFINE: int = 41
    CLASS_DEFINE: int = 42
    EVENT_DEFINE: int = 43
    IMPORT: int = 44
    AND: int = 45
    OR: int = 46
    NOT: int = 47
    RETURN: int = 48
    GREATER: int = 49
    LESS: int = 50
    DOUBLE_COLON: int = 51
    CYCLE_THING: int = 52
    EOF: int = 53
    ELIF: int = 54
    BRACKET_DEFINE: int = 55
    AS: int = 56
    DECORATOR: int = 57
    LESS_OR_EQUALS: int = 58
    MINUS_ASSIGN: int = 59
    PLUS_ASSIGN: int = 60
    EQUALS: int = 61
    DIVIDE_ASSIGN: int = 62
    PR_ASSIGN: int = 63
    DEG_ASSIGN: int = 64
    GREATER_OR_EQUALS: int = 65
    NOT_EQUALS: int = 66
    MULTIPLY_ASSIGN: int = 67
    QUESTION_MARK: int = 68
    DOUBLE_MULTIPLY: int = 69
    NEXT_LINE: int = 70
    JMCC_DEFINE: int = 71
    JMCC_VARIABLE: int = 72
    IN: int = 73
def find_value_from_list(val, possible_list=None):
    if possible_list is None:
        possible_list = set()
    check_enum = 0
    ret_val = val
    for i1 in possible_list:
        if i1.lower().startswith(val.lower()):
            check_enum += 1
            ret_val = i1
        if i1.lower() == val.lower():
            return i1

    if check_enum != 1:
        return -1
    return ret_val

class Token:
    __slots__ = ("type", "value", "starting_pos", "ending_pos", "source", "giga")

    def __init__(self, token_type: int, token_value: str, starting_pos: int, ending_pos: int, source: str, giga=None):
        self.type = token_type
        self.value = token_value
        self.starting_pos = starting_pos
        self.ending_pos = ending_pos
        self.source = source
        self.giga = giga

    def __str__(self):
        return f"Token({self.type},\"{self.value}\")"

    def __repr__(self):
        return self.__str__()

class Lexer:
    __slots__ = ("current_pos", "current_char", "allow_jmcc", "source", "text")

    def __init__(self, txt, source, allow_jmcc=False):
        self.current_pos = -1
        self.current_char = ""
        self.source = source
        self.text = txt
        self.allow_jmcc = allow_jmcc
        self.advance()

    def advance(self):
        self.current_pos += 1
        if self.current_pos < len(self.text):
            self.current_char = self.text[self.current_pos]
        else:
            self.current_char = ""
        if self.current_char == "\r":
            self.current_char = "\n"

    @property
    def next_token(self) -> Token:
        while self.current_char == " ":
            self.advance()
        starting_pos = self.current_pos
        sign_mode = False
        token_value = ""
        if self.current_char.isalpha() or self.current_char == "_":
            while self.current_char != "" and (
                    self.current_char.isalpha() or self.current_char.isdigit() or self.current_char == "_"):
                token_value += self.current_char
                self.advance()
            token_type = None
            match token_value:
                case "if":
                    token_type = Tokens.IF
                case "else":
                    token_type = Tokens.ELSE
                case "function":
                    token_type = Tokens.FUNCTION_DEFINE
                case "def":
                    token_type = Tokens.FUNCTION_DEFINE
                case "fun":
                    token_type = Tokens.FUNCTION_DEFINE
                case "process":
                    token_type = Tokens.PROCESS_DEFINE
                case "var":
                    token_type = Tokens.VAR_DEFINE
                case "inline":
                    token_type = Tokens.INLINE_DEFINE
                case "jmcc":
                    token_type = Tokens.JMCC_DEFINE
                case "bracket":
                    token_type = Tokens.BRACKET_DEFINE
                case "local":
                    token_type = Tokens.LOCAL_DEFINE
                case "game":
                    token_type = Tokens.GAME_DEFINE
                case "save":
                    token_type = Tokens.SAVE_DEFINE
                case "class":
                    token_type = Tokens.CLASS_DEFINE
                case "event":
                    token_type = Tokens.EVENT_DEFINE
                case "import":
                    token_type = Tokens.IMPORT
                case "and":
                    token_type = Tokens.AND
                case "or":
                    token_type = Tokens.OR
                case "not":
                    token_type = Tokens.NOT
                case "return":
                    token_type = Tokens.RETURN
                case "elif":
                    token_type = Tokens.ELIF
                case "as":
                    token_type = Tokens.AS
                case "in":
                    token_type = Tokens.IN
            if token_type is not None:
                return Token(token_type, token_value, starting_pos, self.current_pos - 1, self.source)
            sign_mode = True

        if self.current_char == "\"" or self.current_char == "\'" or self.current_char == "`" or self.current_char == "<":
            mode = self.current_char
            if sign_mode:
                if mode == "`":
                    dect = {"bracket": Tokens.BRACKET_VARIABLE, "inline": Tokens.INLINE_VARIABLE,
                            "local": Tokens.LOCAL_VARIABLE, "game": Tokens.GAME_VARIABLE, "save": Tokens.SAVE_VARIABLE,
                            "jmcc": Tokens.JMCC_VARIABLE}
                    res = find_value_from_list(token_value, dect)
                elif mode in {'"', "'"}:
                    dect = {"plain": Tokens.PLAIN_STRING, "minimessage": Tokens.MINIMESSAGE_STRING,
                            "legacy": Tokens.LEGACY_STRING, "json": Tokens.JSON_STRING}
                    res = find_value_from_list(token_value, dect)
                else:
                    dect = {}
                    res = -1
                if res == -1:
                    return Token(Tokens.VARIABLE, token_value, starting_pos, self.current_pos - 1, self.source)
                token_type = dect[res]
            else:
                if mode == "`":
                    token_type = Tokens.VARIABLE
                elif mode == "<":
                    mode = ">"
                    token_type = Tokens.SUBSTRING
                else:
                    token_type = Tokens.STRING
            self.advance()
            if mode == ">" and self.current_char == "=":
                self.advance()
                return Token(Tokens.LESS_OR_EQUALS, "<=", starting_pos, self.current_pos - 1, self.source)
            if mode == ">" and self.current_char in {" ", "\n"}:
                self.advance()
                return Token(Tokens.LESS, "<", starting_pos, self.current_pos - 1, self.source)
            giga_token = []
            token_value = ""
            block_next_symbol = False
            while self.current_char != "" and (self.current_char != mode or block_next_symbol is True):
                if block_next_symbol:
                    block_next_symbol = False
                    if self.current_char == "n":
                        token_value += "\\"
                    if self.current_char not in {"\n", "\t"}:
                        token_value += self.current_char
                elif self.current_char == "\\":
                    block_next_symbol = True
                elif self.current_char in {"\n", "\t"}:
                    token_value += "\\n"
                elif self.current_char == "$":
                    if len(token_value) > 0:
                        giga_token.append(
                            [Token(Tokens.STRING, token_value, starting_pos, self.current_pos - 1, self.source),
                             Token(Tokens.NONE, "${", self.current_pos, self.current_pos, self.source)])
                    token_value = ""
                    new_starting_pos = self.current_pos
                    self.advance()
                    var_value = ""
                    if self.current_char == "{":
                        self.advance()
                        thing = []
                        counter = 1
                        while self.current_char != "" and not (self.current_char == "}" and counter == 1):
                            if self.current_char == "{":
                                counter += 1
                            elif self.current_char == "}":
                                counter -= 1
                            a = self.next_token
                            if a.type != Tokens.EOF:
                                thing.append(a)
                        if self.current_char == "}":
                            self.advance()
                        if len(thing) > 0:
                            thing.append(Token(Tokens.NONE, "}", self.current_pos, self.current_pos, self.source))
                            giga_token.append(thing)
                    else:
                        while self.current_char != "" and (
                                self.current_char.isalpha() or self.current_char.isdigit() or self.current_char == "_"):
                            var_value += self.current_char
                            self.advance()
                        if len(var_value) > 0:
                            giga_token.append([
                                Token(Tokens.VARIABLE, var_value, new_starting_pos, self.current_pos - 1, self.source),
                                Token(Tokens.NONE, "}", self.current_pos, self.current_pos, self.source)])
                        else:
                            token_value = "$"
                    continue
                else:
                    token_value += self.current_char
                self.advance()
            self.advance()
            if len(giga_token) == 0:
                return Token(token_type, token_value, starting_pos,
                                self.current_pos - 1, self.source)
            else:
                if len(token_value) > 0:
                    giga_token.append([
                        Token(Tokens.STRING, token_value, starting_pos, self.current_pos - 1, self.source),
                        Token(Tokens.NONE, "}", self.current_pos, self.current_pos, self.source)])
                release_index = global_text[self.source].index(mode if mode != ">" else "<", starting_pos,
                                                                self.current_pos - 1) + 1
                release = global_text[self.source][release_index:self.current_pos - 1]
                return Token(token_type, release, starting_pos, self.current_pos - 1, self.source, giga=giga_token)

        if self.current_char == "{":
            if sign_mode:
                if token_value in {"minecraft_nbt", "nbt", "m", "n"}:
                    self.advance()
                    count = 1
                    token_value = "{"
                    while self.current_char != "" and (count != 0):
                        token_value += self.current_char
                        if self.current_char == "{":
                            count += 1
                        elif self.current_char == "}":
                            count -= 1
                        self.advance()
                    return Token(Tokens.SNBT, token_value, starting_pos, self.current_pos - 1, self.source)
                else:
                    return Token(Tokens.VARIABLE, token_value, starting_pos, self.current_pos - 1, self.source)

            self.advance()
            return Token(Tokens.LCPAREN, "{", starting_pos, starting_pos, self.source)

        if sign_mode:
            return Token(Tokens.VARIABLE, token_value, starting_pos, self.current_pos - 1, self.source)
        if self.current_char.isdigit() or self.current_char == "-" or self.current_char == "+":
            minus = False
            plus = False
            es = False
            dot = False
            if self.current_char == "-":
                minus = True
            if self.current_char == "+":
                plus = True
            starting_pos = self.current_pos
            token_value = self.current_char
            self.advance()
            while self.current_char != "" and self.current_char.isdigit() or (
                    es is False and self.current_char == "e") or (
                    dot is False and self.current_char == "." and es is False) or (self.current_char == "_"):
                if es is False and self.current_char == "e":
                    es = True
                if dot is False and self.current_char == "." and es is False:
                    dot = True
                if self.current_char == "_":
                    self.advance()
                    continue
                token_value += self.current_char
                self.advance()
            if token_value == "-":
                if self.current_char == ">":
                    self.advance()
                    return Token(Tokens.CYCLE_THING, "->", starting_pos, self.current_pos - 1, self.source)
                if self.current_char == "=":
                    self.advance()
                    return Token(Tokens.MINUS_ASSIGN, "-=", starting_pos, self.current_pos - 1, self.source)
                return Token(Tokens.MINUS, "-", starting_pos, starting_pos, self.source)
            elif token_value == "+":
                if self.current_char == "=":
                    self.advance()
                    return Token(Tokens.PLUS_ASSIGN, "+=", starting_pos, self.current_pos - 1, self.source)
                return Token(Tokens.PLUS, "+", starting_pos, starting_pos, self.source)
            try:
                return Token(Tokens.PLUS_NUMBER if plus else (Tokens.MINUS_NUMBER if minus else Tokens.NUMBER),
                             int(token_value),
                             starting_pos, self.current_pos - 1, self.source)
            except Exception:
                return Token(Tokens.PLUS_NUMBER if plus else (Tokens.MINUS_NUMBER if minus else Tokens.NUMBER),
                             float(token_value),
                             starting_pos, self.current_pos - 1, self.source)

        if self.current_char == "(":
            self.advance()
            return Token(Tokens.LPAREN, ")", starting_pos, starting_pos, self.source)
        if self.current_char == ")":
            self.advance()
            return Token(Tokens.RPAREN, ")", starting_pos, starting_pos, self.source)
        if self.current_char == ",":
            self.advance()
            return Token(Tokens.COMMA, ",", starting_pos, starting_pos, self.source)
        if self.current_char == ":":
            self.advance()
            if self.current_char == ":":
                self.advance()
                return Token(Tokens.DOUBLE_COLON, "::", starting_pos, self.current_pos - 1, self.source)
            return Token(Tokens.COLON, ":", starting_pos, starting_pos, self.source)
        if self.current_char == ".":
            self.advance()
            return Token(Tokens.DOT, ".", starting_pos, starting_pos, self.source)
        if self.current_char == ";":
            self.advance()
            return Token(Tokens.SEMICOLON, ";", starting_pos, starting_pos, self.source)
        if self.current_char == "[":
            self.advance()
            return Token(Tokens.LSPAREN, "[", starting_pos, starting_pos, self.source)
        if self.current_char == "]":
            self.advance()
            return Token(Tokens.RSPAREN, "]", starting_pos, starting_pos, self.source)
        if self.current_char == "}":
            self.advance()
            return Token(Tokens.RCPAREN, "}", starting_pos, starting_pos, self.source)
        if self.current_char == "=":
            self.advance()
            if self.current_char == "=":
                self.advance()
                return Token(Tokens.EQUALS, "==", starting_pos, self.current_pos - 1, self.source)
            return Token(Tokens.ASSIGN, "=", starting_pos, starting_pos, self.source)
        if self.current_char == "*":
            self.advance()
            if self.current_char == "*":
                self.advance()
                return Token(Tokens.DOUBLE_MULTIPLY, "**", starting_pos, starting_pos + 1, self.source)
            return Token(Tokens.MULTIPLY, "*", starting_pos, starting_pos, self.source)
        if self.current_char == "/":
            self.advance()
            if self.current_char == "/":
                while self.current_char != "":
                    if self.current_char == "\n" or self.current_char == "\t":
                        return self.next_token
                    self.advance()
                return Token(Tokens.EOF, "None", starting_pos, starting_pos, self.source)
            elif self.current_char == "*":
                while self.current_char != "":
                    if self.current_char == "*":
                        self.advance()
                        if self.current_char == "/":
                            self.advance()
                            return self.next_token
                    self.advance()
                return Token(Tokens.EOF, "None", starting_pos, starting_pos, self.source)
            elif self.current_char == "=":
                self.advance()
                return Token(Tokens.DIVIDE_ASSIGN, "/=", starting_pos, self.current_pos - 1, self.source)
            return Token(Tokens.DIVIDE, "/", starting_pos, starting_pos, self.source)
        if self.current_char == "%":
            self.advance()
            if self.current_char == "=":
                self.advance()
                return Token(Tokens.PR_ASSIGN, "%=", starting_pos, self.current_pos - 1, self.source)
            return Token(Tokens.PR, "%", starting_pos, starting_pos, self.source)
        if self.current_char == "^":
            self.advance()
            if self.current_char == "=":
                self.advance()
                return Token(Tokens.DEG_ASSIGN, "^=", starting_pos, self.current_pos - 1, self.source)
            return Token(Tokens.DEG, "^", starting_pos, starting_pos, self.source)
        if self.current_char == ">":
            self.advance()
            if self.current_char == "=":
                self.advance()
                return Token(Tokens.GREATER_OR_EQUALS, ">=", starting_pos, self.current_pos - 1, self.source)
            return Token(Tokens.GREATER, ">", starting_pos, starting_pos, self.source)
        if self.current_char == "!":
            self.advance()
            if self.current_char == "=":
                self.advance()
                return Token(Tokens.NOT_EQUALS, "!=", starting_pos, self.current_pos - 1, self.source)
            return Token(Tokens.NOT, ">", starting_pos, starting_pos, self.source)
        if self.current_char == "@":
            self.advance()
            return Token(Tokens.DECORATOR, "@", starting_pos, starting_pos, self.source)
        if self.current_char == "&":
            self.advance()
            return Token(Tokens.AND, "and", starting_pos, starting_pos, self.source)
        if self.current_char == "|":
            self.advance()
            return Token(Tokens.OR, "or", starting_pos, starting_pos, self.source)
        if self.current_char == "?":
            self.advance()
            return Token(Tokens.QUESTION_MARK, "?", starting_pos, starting_pos, self.source)
        if self.current_char == "\\":
            self.advance()
            while self.current_char == " ":
                self.advance()
            if self.current_char in {"\n", "\t"}:
                self.advance()
                return self.next_token
            else:
                return self.next_token
            
        if self.current_char == "\n" or self.current_char == "\t":
            self.advance()
            return Token(Tokens.NEXT_LINE, "\n", starting_pos, starting_pos, self.source)
        if self.current_char == "":
            return Token(Tokens.EOF, "None", starting_pos, starting_pos, self.source)
        else:
            self.advance()
            return self.next_token

    def get_remaining_tokens(self) -> list:
        lest = []
        while (token := self.next_token).type != Tokens.EOF:
            lest.append(token)
        return lest


def tokenize(txt, source=None, allow_jmcc=False):
    if source is None:
        source = new("source")
    if source not in global_text:
        global_text[source] = txt
    if source not in global_tokens:
        tokens = Lexer(txt, source, allow_jmcc=allow_jmcc).get_remaining_tokens()
        global_tokens[source] = tokens
    return global_tokens[source]

def clear(source):
    if source is None:
        return
    if source in global_text:
        del global_text[source]
    if source in global_tokens:
        del global_tokens[source]

def line_and_offset_to_pos(text: str, line, offset):
    start_pos = -1
    while line > 0:
        start_pos = text.find("\n", start_pos + 1)
        line -= 1
    return start_pos + 1 + offset
def pos_to_line_and_offset(txt:str, starting_pos, ending_pos):
    start_line_index = txt.rfind("\n", 0, starting_pos)
    start_line = txt.count("\n", 0, starting_pos)
    ending_line_index = txt.find("\n", ending_pos)
    end_line = txt.count("\n", 0, ending_pos)
    if ending_line_index == -1:
        ending_line_index = len(txt)
    if start_line_index == -1:
        start_line_index = 0
    elif start_line_index > 0:
        start_line_index += 1
    return (start_line, start_line_index), (end_line, ending_line_index)

def find_token_that_have_pos(source, pos):
    for i1 in range(len(global_tokens[source])):
        i = global_tokens[source][i1]
        i: Token
        if i.starting_pos <= pos <= i.ending_pos:
            return i1
    return i1

def try_find_object(source, pos):
    if pos + 1 < len(global_tokens[source]) and pos >= 1 and global_tokens[source][pos-1].type == Tokens.VARIABLE and global_tokens[source][pos].type == Tokens.DOUBLE_COLON and global_tokens[source][pos+1].type == Tokens.VARIABLE:
        return global_tokens[source][pos-1].value + global_tokens[source][pos].value + global_tokens[source][pos+1].value
    if global_tokens[source][pos].type == Tokens.VARIABLE:
        if pos >= 1 and global_tokens[source][pos-1].type == Tokens.DOT:
            return "."+global_tokens[source][pos].value
        if pos >= 2 and global_tokens[source][pos-2].type == Tokens.VARIABLE and global_tokens[source][pos-1].type == Tokens.DOUBLE_COLON:
            return global_tokens[source][pos-2].value + global_tokens[source][pos-1].value + global_tokens[source][pos].value
        if pos + 2 < len(global_tokens[source]) and global_tokens[source][pos+1].type == Tokens.DOUBLE_COLON and global_tokens[source][pos+2].type == Tokens.VARIABLE:
            return global_tokens[source][pos].value + global_tokens[source][pos+1].value + global_tokens[source][pos+2].value
    if pos + 1 < len(global_tokens[source]) and global_tokens[source][pos].type == Tokens.EVENT_DEFINE and global_tokens[source][pos+1].type == Tokens.SUBSTRING:
        return global_tokens[source][pos].value + "<" + global_tokens[source][pos+1].value + ">"
    if pos >= 1 and global_tokens[source][pos].type == Tokens.SUBSTRING and global_tokens[source][pos-1].type == Tokens.EVENT_DEFINE:
        return global_tokens[source][pos-1].value + "<" + global_tokens[source][pos].value + ">"

    return str(global_tokens[source][pos].value)