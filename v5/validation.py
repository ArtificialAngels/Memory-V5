"""统一校验框架 (Unified validation framework) for Ikaros V5.

Provides input checking, logic verification, error classification,
and extensible rule registration.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from enum import Enum
from functools import wraps
from typing import Any, Callable, ClassVar

logger = logging.getLogger("ikaros.v5.validation")

# ---------------------------------------------------------------------------
# Error codes (stable, never reuse deprecated codes)
# ---------------------------------------------------------------------------


class ErrorCode(str, Enum):
    """Stable error codes. Deprecate by renaming to _DEPRECATED_xxx, never delete."""

    # Input validation (Vx-01xx)
    IN_EMPTY_CONTENT = "V5-0101"
    IN_CONTENT_TOO_LONG = "V5-0102"
    IN_INVALID_TYPE = "V5-0103"
    IN_WEIGHT_OUT_OF_RANGE = "V5-0104"
    IN_INVALID_PAD = "V5-0105"
    IN_TAGS_TOO_LONG = "V5-0106"
    IN_EMPTY_QUERY = "V5-0107"
    IN_QUERY_TOO_SHORT = "V5-0108"
    IN_STRUCTURED_MALFORMED = "V5-0109"

    # Logic verification (Vx-02xx)
    LG_DUPLICATE_MEMORY = "V5-0201"
    LG_CONFLICTING_TAGS = "V5-0202"
    LG_STALE_DATA = "V5-0203"
    LG_INCONSISTENT_STATE = "V5-0204"
    LG_CIRCULAR_REFERENCE = "V5-0205"
    LG_ENTITY_NOT_FOUND = "V5-0206"

    # System integrity (Vx-03xx)
    SY_DB_CONNECTION = "V5-0301"
    SY_DISK_FULL = "V5-0302"
    SY_CONFIG_MISSING = "V5-0303"


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ValidationError:
    code: ErrorCode
    message: str
    severity: Severity = Severity.WARNING
    field_name: str = ""
    actual_value: Any = None
    detail: dict[str, Any] = dc_field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity.value,
            "field": self.field_name,
            "value": str(self.actual_value) if self.actual_value is not None else None,
            "detail": self.detail,
        }


class ValidationRule(ABC):
    name: ClassVar[str] = "base"
    description: ClassVar[str] = "Base validation rule"

    enabled: bool = True

    @abstractmethod
    def validate(self, value: Any, context: dict[str, Any] | None = None) -> list[ValidationError]:
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ValidationRegistry:
    def __init__(self) -> None:
        self._rules: dict[str, ValidationRule] = {}
        self._disabled: set[str] = set()

    def register(self, rule: ValidationRule) -> None:
        self._rules[rule.name] = rule

    def disable(self, name: str) -> None:
        self._disabled.add(name)

    def enable(self, name: str) -> None:
        self._disabled.discard(name)

    def is_enabled(self, name: str) -> bool:
        return name not in self._disabled

    def get(self, name: str) -> ValidationRule | None:
        return self._rules.get(name)

    def apply(self, value: Any, context: dict[str, Any] | None = None) -> list[ValidationError]:
        errors: list[ValidationError] = []
        for name, rule in self._rules.items():
            if name in self._disabled:
                continue
            try:
                results = rule.validate(value, context)
                errors.extend(results)
            except Exception as exc:
                logger.warning("validation: rule %s raised %s, skipping", name, exc)
        return errors

    def apply_or_raise(
        self, value: Any, context: dict[str, Any] | None = None,
        min_severity: Severity = Severity.ERROR
    ) -> list[ValidationError]:
        errors = self.apply(value, context)
        critical = [e for e in errors if _severity_rank(e.severity) >= _severity_rank(min_severity)]
        if critical:
            raise ValidationFailed(critical)
        return errors

    @property
    def rule_names(self) -> list[str]:
        return sorted(self._rules.keys())

    def summary(self) -> dict[str, Any]:
        return {
            "total_rules": len(self._rules),
            "enabled": len(self._rules) - len(self._disabled),
            "disabled": sorted(self._disabled),
            "rule_names": self.rule_names,
        }


class ValidationFailed(Exception):
    def __init__(self, errors: list[ValidationError]) -> None:
        self.errors = errors
        codes = ", ".join(e.code.actual_value for e in errors)
        messages = "; ".join(e.message for e in errors)
        super().__init__(f"Validation failed [{codes}]: {messages}")


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------


class NotEmptyRule(ValidationRule):
    name = "not_empty"
    description = "Value must not be empty or whitespace-only"

    def validate(self, value: Any, context: dict[str, Any] | None = None) -> list[ValidationError]:
        if not value or (isinstance(value, str) and not value.strip()):
            return [
                ValidationError(
                    code=ErrorCode.IN_EMPTY_CONTENT,
                    message="Value must not be empty",
                    field_name=context.get("field", "input") if context else "input",
                )
            ]
        return []


class MaxLengthRule(ValidationRule):
    name = "max_length"
    description = "String must not exceed max_length characters"

    def validate(self, value: Any, context: dict[str, Any] | None = None) -> list[ValidationError]:
        ctx = context or {}
        max_len = ctx.get("max_length", 10000)
        if isinstance(value, str) and len(value) > max_len:
            return [
                ValidationError(
                    code=ErrorCode.IN_CONTENT_TOO_LONG,
                    message=f"Value exceeds maximum length of {max_len} (got {len(value)})",
                    field_name=ctx.get("field", "input"),
                    detail={"max_length": max_len, "actual": len(value)},
                )
            ]
        return []


class AllowedTypeRule(ValidationRule):
    name = "allowed_type"
    description = "Memory type must be one of the allowed values"

    ALLOWED: ClassVar[set[str]] = {
        "fact", "conversation", "emotion", "reflection",
        "event", "task", "thought", "dream",
        "emotional_event", "emotion_label", "activity_reflection",
        "identity", "preference", "lesson", "milestone",
        "decision", "dissonance", "narrative", "philosophy",
        "audit", "self_discovery", "self_reflection",
        "inner_monologue", "thought_marker", "technological_discovery",
        "repair_test", "test", "user_trait",
    }

    def validate(self, value: Any, context: dict[str, Any] | None = None) -> list[ValidationError]:
        ctx = context or {}
        mem_type = ctx.get("type", "") or (value if isinstance(value, str) else "")
        if mem_type and mem_type not in self.ALLOWED:
            return [
                ValidationError(
                    code=ErrorCode.IN_INVALID_TYPE,
                    message=f"Memory type '{mem_type}' is not allowed. Allowed: {sorted(self.ALLOWED)}",
                    field_name="type",
                    actual_value=mem_type,
                    detail={"allowed": sorted(self.ALLOWED)},
                )
            ]
        return []


class WeightRangeRule(ValidationRule):
    name = "weight_range"
    description = "Weight must be between 0.0 and 1.0"

    def validate(self, value: Any, context: dict[str, Any] | None = None) -> list[ValidationError]:
        weight = value if isinstance(value, (int, float)) else (
            context.get("weight", 0.6) if context else 0.6
        )
        try:
            w = float(weight)
            if w < 0.0 or w > 1.0:
                return [
                    ValidationError(
                        code=ErrorCode.IN_WEIGHT_OUT_OF_RANGE,
                        message=f"Weight must be between 0.0 and 1.0 (got {w})",
                        field_name="weight",
                        actual_value=w,
                    )
                ]
        except (ValueError, TypeError):
            return [
                ValidationError(
                    code=ErrorCode.IN_WEIGHT_OUT_OF_RANGE,
                    message=f"Weight must be a number (got {weight})",
                    field_name="weight",
                    actual_value=weight,
                )
            ]
        return []


class PADRangeRule(ValidationRule):
    name = "pad_range"
    description = "PAD values must be between -1.0 and 1.0"

    def validate(self, value: Any, context: dict[str, Any] | None = None) -> list[ValidationError]:
        ctx = context or {}
        errors: list[ValidationError] = []
        for dim in ("pad_p", "pad_a", "pad_d"):
            val = ctx.get(dim, 0.0)
            try:
                v = float(val)
                if v < -1.0 or v > 1.0:
                    errors.append(
                        ValidationError(
                            code=ErrorCode.IN_INVALID_PAD,
                            message=f"PAD {dim} must be between -1.0 and 1.0 (got {v})",
                            field_name=dim,
                            actual_value=v,
                        )
                    )
            except (ValueError, TypeError):
                pass
        return errors


class QueryMinLengthRule(ValidationRule):
    name = "query_min_length"
    description = "Search query must be at least 2 characters"

    def validate(self, value: Any, context: dict[str, Any] | None = None) -> list[ValidationError]:
        if isinstance(value, str) and len(value.strip()) < 2:
            return [
                ValidationError(
                    code=ErrorCode.IN_QUERY_TOO_SHORT,
                    message="Search query must be at least 2 characters",
                    field_name="query",
                )
            ]
        return []


class EntityTypeRule(ValidationRule):
    name = "entity_type"
    description = "Entity type must be person/place/object/event"

    ALLOWED: ClassVar[set[str]] = {"person", "place", "object", "event"}

    def validate(self, value: Any, context: dict[str, Any] | None = None) -> list[ValidationError]:
        etype = value if isinstance(value, str) else (context or {}).get("entity_type", "")
        if etype and etype not in self.ALLOWED:
            return [
                ValidationError(
                    code=ErrorCode.IN_INVALID_TYPE,
                    message=f"Entity type '{etype}' not in {sorted(self.ALLOWED)}",
                    field_name="entity_type",
                    actual_value=etype,
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Global registries
# ---------------------------------------------------------------------------

_input_registry = ValidationRegistry()
_memory_registry = ValidationRegistry()
_query_registry = ValidationRegistry()
_entity_registry = ValidationRegistry()


def _init_registries():
    """Register all built-in rules. Idempotent."""

    if _input_registry.rule_names:
        return

    # Input validators
    _input_registry.register(NotEmptyRule())
    _input_registry.register(MaxLengthRule())

    # Memory validators
    _memory_registry.register(NotEmptyRule())
    _memory_registry.register(MaxLengthRule())
    _memory_registry.register(AllowedTypeRule())
    _memory_registry.register(WeightRangeRule())
    _memory_registry.register(PADRangeRule())

    # Query validators
    _query_registry.register(NotEmptyRule())
    _query_registry.register(QueryMinLengthRule())
    _query_registry.register(MaxLengthRule())

    # Entity validators
    _entity_registry.register(NotEmptyRule())
    _entity_registry.register(EntityTypeRule())


_init_registries()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_validation_config() -> dict[str, Any]:
    """Load validation config from preprocess_config.yaml (validation: section)."""
    try:
        from v5 import preprocess_config as pc
        cfg = pc.cfg()
        return cfg.get("validation", {})
    except Exception:
        return {}


def apply_config() -> None:
    """Apply validation config: enable/disable rules per registry."""
    config = load_validation_config()
    for section, registry in [
        ("input", _input_registry),
        ("memory", _memory_registry),
        ("query", _query_registry),
        ("entity", _entity_registry),
    ]:
        section_cfg = config.get(section, {})
        for rule_name, rule_cfg in section_cfg.items():
            if isinstance(rule_cfg, dict):
                enabled = rule_cfg.get("enabled", True)
                if not enabled:
                    registry.disable(rule_name)
            elif isinstance(rule_cfg, bool) and not rule_cfg:
                registry.disable(rule_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_input(value: Any, field: str = "input",
                   max_length: int = 10000) -> list[ValidationError]:
    return _input_registry.apply(value, {"field": field, "max_length": max_length})


def validate_memory(content: str, mem_type: str = "fact",
                    weight: float = 0.6, pad_p: float = 0.0,
                    pad_a: float = 0.0, pad_d: float = 0.0,
                    max_length: int = 10000) -> list[ValidationError]:
    return _memory_registry.apply(content, {
        "field": "content",
        "type": mem_type,
        "weight": weight,
        "pad_p": pad_p, "pad_a": pad_a, "pad_d": pad_d,
        "max_length": max_length,
    })


def validate_query(query: str, max_length: int = 500) -> list[ValidationError]:
    return _query_registry.apply(query, {
        "field": "query",
        "max_length": max_length,
    })


def validate_entity(entity_type: str, name: str = "") -> list[ValidationError]:
    return _entity_registry.apply(entity_type, {
        "field": "entity_type",
        "entity_type": entity_type,
        "name": name,
    })


# ---------------------------------------------------------------------------
# Structured pipeline guard (Task #15)
# ---------------------------------------------------------------------------
#
# The consolidate / distill / reflect pipelines ask the cloud LLM for a clean
# structured statement (a fact / preference / lesson / user_trait) and persist
# the parsed ``content``. The cloud model occasionally returns a conversational
# preamble ("Okay, the user wants..."), leaks raw JSON or a markdown fence
# (```), or dumps a runaway block of text. None of those belong in the memory
# DB. This guard flags such content so the pipeline can skip persisting it.


class StructuredContentGuard:
    """Flag LLM narration / malformed output that must NOT be stored through
    the structured pipeline (consolidate / distill / reflect)."""

    # Preambles that indicate the model narrated instead of answering.
    # English bare prefixes are the documented failure mode; Chinese prefixes
    # require trailing punctuation so a real fact starting with the same word
    # (e.g. "当然要相信哥哥") is not falsely flagged.
    NARRATION_PREFIXES = (
        "okay", "ok", "sure", "alright", "got it", "here", "below",
        "i am", "let me", "certainly", "sure,",
        "好的，", "好的,", "当然，", "当然,", "当然可以",
        "以下是", "下面是", "作为", "我是一个",
    )
    # Raw model artifacts that must never appear in a stored content field.
    FORBIDDEN_SUBSTRINGS = ("```",)
    # Structured pipeline statements are concise; beyond this is a raw dump.
    MAX_STRUCTURED_LEN = 800

    @classmethod
    def guard(cls, content: str) -> list[ValidationError]:
        errors: list[ValidationError] = []
        if not isinstance(content, str):
            errors.append(ValidationError(
                code=ErrorCode.IN_STRUCTURED_MALFORMED,
                message="Structured content must be a string",
                severity=Severity.WARNING,
            ))
            return errors
        c = content.strip()
        if not c:
            return errors  # 空由 NotEmptyRule 处理, 这里不重复报
        low = c.lower()
        # 1) narration preamble at the very start
        for p in cls.NARRATION_PREFIXES:
            if low.startswith(p):
                errors.append(ValidationError(
                    code=ErrorCode.IN_STRUCTURED_MALFORMED,
                    message=f"内容以旁白前缀开头, 疑似 LLM 旁白而非结构化陈述: {p!r}",
                    severity=Severity.WARNING,
                    detail={"prefix": p},
                ))
                break
        # 2) leaked raw JSON / markdown fence
        if c.startswith("{") or c.startswith("["):
            errors.append(ValidationError(
                code=ErrorCode.IN_STRUCTURED_MALFORMED,
                message="内容以 { 或 [ 开头, 疑似未解析的 JSON 被直接落库",
                severity=Severity.WARNING,
            ))
        for sub in cls.FORBIDDEN_SUBSTRINGS:
            if sub in content:
                errors.append(ValidationError(
                    code=ErrorCode.IN_STRUCTURED_MALFORMED,
                    message=f"内容含非法字符 {sub!r}, 疑似 markdown fence 未清理",
                    severity=Severity.WARNING,
                ))
                break
        # 3) runaway length (raw dump)
        if len(c) > cls.MAX_STRUCTURED_LEN:
            errors.append(ValidationError(
                code=ErrorCode.IN_STRUCTURED_MALFORMED,
                message=f"结构化内容过长 ({len(c)} > {cls.MAX_STRUCTURED_LEN}), 疑似整段模型输出被落库",
                severity=Severity.WARNING,
                detail={"length": len(c)},
            ))
        return errors


def guard_structured_content(content: str) -> list[ValidationError]:
    """Validate a single structured-pipeline content string.

    Returns a list of ValidationError (possibly empty). Callers in the
    consolidate/distill/reflect pipeline should skip persisting any content
    that yields errors here, to avoid polluting the memory DB with LLM
    narration or malformed output.
    """
    return StructuredContentGuard.guard(content)


def is_clean_structured_content(content: str) -> bool:
    """Convenience: True if the content is safe to persist via the
    structured pipeline."""
    return not guard_structured_content(content)


def check_and_log(value: Any, validator: Callable[[Any], list[ValidationError]],
                  context: str = "") -> bool:
    errors = validator(value)
    for err in errors:
        log_fn = {
            Severity.DEBUG: logger.debug,
            Severity.INFO: logger.info,
            Severity.WARNING: logger.warning,
            Severity.ERROR: logger.error,
            Severity.CRITICAL: logger.critical,
        }.get(err.severity, logger.warning)
        log_fn("validation [%s] %s: %s (field=%s value=%s)",
               context, err.code.value, err.message, err.field_name, err.actual_value)
    return len(errors) == 0


def registry_summary() -> dict[str, Any]:
    return {
        "input": _input_registry.summary(),
        "memory": _memory_registry.summary(),
        "query": _query_registry.summary(),
        "entity": _entity_registry.summary(),
    }


def register_custom_rule(category: str, rule: ValidationRule) -> None:
    registry = {
        "input": _input_registry,
        "memory": _memory_registry,
        "query": _query_registry,
        "entity": _entity_registry,
    }.get(category)
    if registry is None:
        raise ValueError(f"Unknown validation category: {category}")
    registry.register(rule)
    logger.info("validation: registered custom rule %s in category %s", rule.name, category)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def validated(validator: Callable[..., list[ValidationError]],
              raise_on: Severity = Severity.ERROR):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Run validation before function execution
            errors = validator(*args, **kwargs)
            for err in errors:
                logger.warning(
                    "validation [%s] %s: %s", func.__name__, err.code.actual_value, err.message
                )
            critical = [e for e in errors if _severity_rank(e.severity) >= _severity_rank(raise_on)]
            if critical:
                raise ValidationFailed(critical)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _severity_rank(sev: Severity) -> int:
    return {Severity.DEBUG: 0, Severity.INFO: 1, Severity.WARNING: 2,
            Severity.ERROR: 3, Severity.CRITICAL: 4}[sev]


# Apply config on import
apply_config()
