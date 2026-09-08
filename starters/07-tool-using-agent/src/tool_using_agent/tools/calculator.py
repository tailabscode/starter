"""calculator — safe arithmetic evaluation.

Deliberately does NOT use `eval`. A small hand-written tokenizer plus a
recursive-descent parser supports exactly: `+ - * / ( ) .` and decimal
numbers. Anything else (names, function calls, exponents, ...) is rejected
with a clear `ToolError` rather than silently doing something unexpected.

Grammar:
    expr   := term (('+' | '-') term)*
    term   := factor (('*' | '/') factor)*
    factor := ('+' | '-') factor | '(' expr ')' | NUMBER
"""

import re

from .base import ToolError

_TOKEN_RE = re.compile(r"\s*(?:(\d+\.\d+|\d+)|([()+\-*/]))")


class CalculatorError(Exception):
    """Raised for a malformed or unsafe expression, or a math error like /0."""


def _tokenize(expression: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    length = len(expression)
    while pos < length:
        match = _TOKEN_RE.match(expression, pos)
        if match is None:
            if expression[pos:].strip() == "":
                break
            bad_char = expression[pos:].strip()[0]
            raise CalculatorError(f"unsupported character {bad_char!r} in expression")
        pos = match.end()
        tokens.append(match.group(1) or match.group(2))
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]):
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> str | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _advance(self) -> str:
        token = self._peek()
        if token is None:
            raise CalculatorError("unexpected end of expression")
        self._pos += 1
        return token

    def parse_expr(self) -> float:
        value = self.parse_term()
        while self._peek() in ("+", "-"):
            op = self._advance()
            rhs = self.parse_term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def parse_term(self) -> float:
        value = self.parse_factor()
        while self._peek() in ("*", "/"):
            op = self._advance()
            rhs = self.parse_factor()
            if op == "/":
                if rhs == 0:
                    raise CalculatorError("division by zero")
                value = value / rhs
            else:
                value = value * rhs
        return value

    def parse_factor(self) -> float:
        token = self._peek()
        if token in ("+", "-"):
            op = self._advance()
            value = self.parse_factor()
            return -value if op == "-" else value
        if token == "(":
            self._advance()
            value = self.parse_expr()
            if self._peek() != ")":
                raise CalculatorError("missing closing parenthesis")
            self._advance()
            return value
        if token is None:
            raise CalculatorError("unexpected end of expression")
        try:
            return float(self._advance())
        except ValueError as e:
            raise CalculatorError(f"invalid number: {token}") from e

    def finish(self) -> None:
        leftover = self._peek()
        if leftover is not None:
            raise CalculatorError(f"unexpected token after expression: {leftover!r}")


def evaluate(expression: str) -> float:
    """Evaluate a restricted arithmetic expression. Raises `CalculatorError` on failure."""
    tokens = _tokenize(expression)
    if not tokens:
        raise CalculatorError("empty expression")
    parser = _Parser(tokens)
    value = parser.parse_expr()
    parser.finish()
    return value


def _format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return str(value)


SCHEMA = {
    "name": "calculator",
    "description": (
        "Evaluate a basic arithmetic expression: addition, subtraction, "
        "multiplication, division, parentheses, and decimal numbers only. "
        "No variables, functions, or exponents. Rejects division by zero."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, e.g. '(3 + 4) * 2'.",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
}


def run(tool_input: dict) -> str:
    expression = tool_input.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("`expression` must be a non-empty string.")
    try:
        result = evaluate(expression)
    except CalculatorError as e:
        raise ToolError(f"Could not evaluate '{expression}': {e}") from e
    return f"{expression} = {_format_number(result)}"
