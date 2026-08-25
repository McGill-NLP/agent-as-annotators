"""Difficulty stratification for Exploration v2.

A single 50-step exploration is sliced into five step bands. A task proposed from
a step in band N draws on everything the explorer learned in the steps before it,
so the bands form a difficulty gradient purely through accumulated site
knowledge. The levels are **independent, not compositional**: an L5 task is not
an L1 task with extra sub-goals bolted on, it is a task proposed from a point in
the trajectory where the explorer knew more about the site.

    L1  steps  0-9      L2  steps 10-19     L3  steps 20-29
    L4  steps 30-39     L5  steps 40-49

``difficulty_level`` is **optional** on a task record. A record that omits it, or
sets it to ``None``, is a pre-v2 record: no level is derived from any other
field, nothing is validated, and no key is added to it anywhere downstream. That
is what keeps already-published A3-Synth data working unchanged.

Step 0 note: the task-intent sampler drops step 0, so L1 in practice draws from
steps 1-9 (nine candidates) while every other band has ten. That is pre-existing
behaviour, recorded here so the L1 band is not mistaken for a clean decile.
"""

from typing import Any, Iterable, Mapping, Optional

# Exploration v2 budget. 50 = 5 bands x 10 steps, so no band is ragged.
EXPLORATION_MAX_STEPS = 50
STEP_BAND_WIDTH = 10
DIFFICULTY_LEVELS = (1, 2, 3, 4, 5)

MIN_DIFFICULTY_LEVEL = DIFFICULTY_LEVELS[0]
MAX_DIFFICULTY_LEVEL = DIFFICULTY_LEVELS[-1]

# Record keys. Both PRs implementing Exploration v2 must agree on these.
DIFFICULTY_LEVEL_KEY = "difficulty_level"
STEP_BAND_KEY = "step_band"
EXPLORATION_STEP_KEY = "exploration_step_num"

# The band arithmetic is the whole point of the 50-step budget, so tie the two
# together mechanically rather than by comment: changing one without the other
# is what produces a ragged final band.
assert EXPLORATION_MAX_STEPS == len(DIFFICULTY_LEVELS) * STEP_BAND_WIDTH, (
    f"EXPLORATION_MAX_STEPS ({EXPLORATION_MAX_STEPS}) must equal "
    f"len(DIFFICULTY_LEVELS) ({len(DIFFICULTY_LEVELS)}) * STEP_BAND_WIDTH "
    f"({STEP_BAND_WIDTH}); otherwise the bands do not tile the trajectory."
)


def _describe(source: Optional[str]) -> str:
    return f" (in {source})" if source else ""


def step_band(level: int) -> tuple[int, int]:
    """Return the inclusive ``(first_step, last_step)`` covered by ``level``."""
    if level not in DIFFICULTY_LEVELS:
        raise ValueError(
            f"difficulty_level must be one of {list(DIFFICULTY_LEVELS)}, got {level!r}"
        )
    first = (level - MIN_DIFFICULTY_LEVEL) * STEP_BAND_WIDTH
    return first, first + STEP_BAND_WIDTH - 1


def level_for_step(step: int) -> int:
    """Return the difficulty level whose band contains ``step``."""
    if not isinstance(step, int) or isinstance(step, bool):
        raise TypeError(f"step must be an int, got {type(step).__name__}: {step!r}")
    if not 0 <= step < EXPLORATION_MAX_STEPS:
        raise ValueError(
            f"step {step} is outside the exploration budget "
            f"[0, {EXPLORATION_MAX_STEPS - 1}]"
        )
    return MIN_DIFFICULTY_LEVEL + step // STEP_BAND_WIDTH


def normalize_difficulty_level(
    value: Any, *, source: Optional[str] = None
) -> Optional[int]:
    """Validate a ``difficulty_level`` value.

    ``None`` passes through as ``None`` -- that is the pre-v2 record and it must
    stay indistinguishable from today. Everything else must be a plain ``int`` in
    ``[1, 5]``. Strings and floats are rejected rather than coerced: a config
    carrying ``"3"`` or ``3.0`` was written by something that does not share this
    schema, and silently accepting it is how a difficulty axis ends up encoding
    whatever the other side happened to mean.
    """
    if value is None:
        return None
    # bool is a subclass of int, so `True` would otherwise validate as level 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{DIFFICULTY_LEVEL_KEY} must be an int in "
            f"[{MIN_DIFFICULTY_LEVEL}, {MAX_DIFFICULTY_LEVEL}] or None, got "
            f"{type(value).__name__}: {value!r}{_describe(source)}"
        )
    if value not in DIFFICULTY_LEVELS:
        raise ValueError(
            f"{DIFFICULTY_LEVEL_KEY} must be one of {list(DIFFICULTY_LEVELS)}, got "
            f"{value!r}{_describe(source)}"
        )
    return value


def normalize_step_band(
    value: Any, *, level: Optional[int] = None, source: Optional[str] = None
) -> Optional[tuple[int, int]]:
    """Validate an optional ``step_band`` value.

    The agreed shape is a two-element ``[first_step, last_step]``, inclusive on
    both ends, matching the ``steps 0-9`` form used in the design table. When
    ``level`` is given the band must be exactly the band that level defines --
    a record cannot claim L2 and carry L4's step range.
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(
            f"{STEP_BAND_KEY} must be a two-element [first_step, last_step] list "
            f"or None, got {type(value).__name__}: {value!r}{_describe(source)}"
        )
    band = tuple(value)
    if len(band) != 2 or any(
        isinstance(edge, bool) or not isinstance(edge, int) for edge in band
    ):
        raise ValueError(
            f"{STEP_BAND_KEY} must be a two-element [first_step, last_step] list of "
            f"ints, got {value!r}{_describe(source)}"
        )
    if level is not None:
        expected = step_band(level)
        if band != expected:
            raise ValueError(
                f"{STEP_BAND_KEY} {list(band)} does not match "
                f"{DIFFICULTY_LEVEL_KEY} {level}, whose band is "
                f"{list(expected)}{_describe(source)}"
            )
    return band


def validate_task_record(
    record: Mapping[str, Any], *, source: Optional[str] = None
) -> Optional[int]:
    """Validate the difficulty fields of one task record and return its level.

    Returns ``None`` for a record with no ``difficulty_level``, which is the
    pre-v2 case and is left completely alone. For a record that does carry one,
    three things are asserted:

    1. the level is an int in ``[1, 5]``;
    2. ``step_band``, if present, is the band that level defines;
    3. ``exploration_step_num``, if present, falls inside that band.

    (3) is the check that catches a mis-stratified dataset. The level is supposed
    to be *derived* from the sampled step, so the two disagreeing means the
    stratification is measuring something other than what it claims to.
    """
    if source is None:
        task_id = record.get("task_id")
        if task_id is not None:
            source = f"task_id={task_id}"

    level = normalize_difficulty_level(record.get(DIFFICULTY_LEVEL_KEY), source=source)
    if level is None:
        # Pre-v2 record. Deliberately do NOT derive a level from
        # exploration_step_num: an absent field must stay absent.
        return None

    normalize_step_band(record.get(STEP_BAND_KEY), level=level, source=source)

    step = record.get(EXPLORATION_STEP_KEY)
    if step is not None:
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError(
                f"{EXPLORATION_STEP_KEY} must be an int or None, got "
                f"{type(step).__name__}: {step!r}{_describe(source)}"
            )
        first, last = step_band(level)
        if not first <= step <= last:
            raise ValueError(
                f"{EXPLORATION_STEP_KEY} {step} falls outside the band "
                f"[{first}, {last}] implied by {DIFFICULTY_LEVEL_KEY} "
                f"{level}{_describe(source)}"
            )

    return level


def task_info_for_level(level: Optional[int]) -> dict[str, int]:
    """Build the ``setup()`` info payload carrying a difficulty level.

    Returns ``{}`` for ``None``, so a pre-v2 task's ``info["task_info"]`` is
    exactly the empty dict it was before Exploration v2 existed rather than
    ``{"difficulty_level": None}`` -- any consumer doing a plain truthiness test
    on it keeps behaving identically.
    """
    return {} if level is None else {DIFFICULTY_LEVEL_KEY: level}


def difficulty_level_tally(
    records: Iterable[Mapping[str, Any]], *, source: Optional[str] = None
) -> dict[Optional[int], int]:
    """Validate every record and count them per level.

    The ``None`` key counts records with no ``difficulty_level``. Levels with
    zero records are reported as ``0`` rather than omitted, so an unfilled band
    is visible in the tally instead of having to be inferred from its absence --
    L5 (steps 40-49) is the one most likely to come back empty, since it only
    fills when explorations actually run the full budget.
    """
    tally: dict[Optional[int], int] = {level: 0 for level in DIFFICULTY_LEVELS}
    tally[None] = 0
    for record in records:
        tally[validate_task_record(record, source=source)] += 1
    return tally


def format_difficulty_level_tally(tally: Mapping[Optional[int], int]) -> str:
    """Render a tally as a single log line."""
    parts = [f"L{level}={tally.get(level, 0)}" for level in DIFFICULTY_LEVELS]
    parts.append(f"unlevelled={tally.get(None, 0)}")
    return ", ".join(parts)
