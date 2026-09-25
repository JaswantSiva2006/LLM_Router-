"""Exact, deliberately small-scope numeric parsing for GSM8K answers."""

from __future__ import annotations

import ast
import operator
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Final

_NUMBER_RE: Final = re.compile(
    r"(?<![\w.])(?P<number>[+-]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?)(?![\w.])"
)

_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
         "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_WORD_TOKEN = re.compile(r"[a-z]+(?:-[a-z]+)?", re.I)


def parse_numeric(value: str) -> Fraction | None:
    """Parse an entire integer, decimal, or simple fraction exactly."""
    text = value.strip().replace("−", "-").replace(",", "")
    text = re.sub(r"^[\s$]+|[\s$!?.;:]+$", "", text)
    latex_fraction = re.fullmatch(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", text)
    if latex_fraction:
        numerator = parse_numeric(latex_fraction.group(1))
        denominator = parse_numeric(latex_fraction.group(2))
        if numerator is None or denominator in {None, Fraction(0)}:
            return None
        return numerator / denominator
    if not text:
        return None
    if text.count("/") == 1:
        left, right = (part.strip() for part in text.split("/"))
        try:
            denominator = Decimal(right)
            if denominator == 0:
                return None
            return Fraction(Decimal(left)) / Fraction(denominator)
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None
    try:
        return Fraction(Decimal(text))
    except (InvalidOperation, ValueError):
        return None


def parse_written_number(value: str) -> Fraction | None:
    """Parse practical English cardinal forms (e.g. 'negative seventy two')."""
    words = []
    for token in _WORD_TOKEN.findall(value.lower()):
        words.extend(token.split("-"))
    if not words:
        return None
    sign = -1 if words and words[0] in {"negative", "minus"} else 1
    if sign == -1:
        words = words[1:]
    words = [word for word in words if word != "and"]
    if not words or any(w not in _ONES and w not in _TENS and w not in _SCALES and w != "hundred" for w in words):
        return None
    total = current = 0
    for word in words:
        if word in _ONES:
            current += _ONES[word]
        elif word in _TENS:
            current += _TENS[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        else:
            total += max(current, 1) * _SCALES[word]
            current = 0
    return Fraction(sign * (total + current))


def numeric_tokens(text: str) -> list[tuple[str, Fraction, int, int]]:
    """Return boundary-safe numeric literals with exact values and spans."""
    result = []
    for match in _NUMBER_RE.finditer(text):
        parsed = parse_numeric(match.group("number"))
        if parsed is not None:
            result.append((match.group("number"), parsed, match.start(), match.end()))
    return result


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def evaluate_arithmetic(expression: str) -> Fraction | None:
    """Safely evaluate a simple literal arithmetic expression; never calls eval."""
    expression = expression.replace("×", "*").replace("÷", "/").replace("−", "-")
    expression = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError
            return _OPS[type(node.op)](left, right)
        raise ValueError

    try:
        return visit(tree)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def fraction_to_string(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
