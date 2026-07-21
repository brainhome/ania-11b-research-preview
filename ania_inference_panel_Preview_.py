# ========================================
# ania_inference_panel_v3.py
# ========================================
"""
Architektura każdej tury:
    1. Prywatna notatka robocza (niewidoczna domyślnie; przełącznik
       "Pokaż myślenie" w bocznym menu wyświetla ją wstecznie).
    2. Finalna odpowiedź wykorzystująca notatkę.

Zasady: BF16 bez kwantyzacji, kontekst 65 536, repetition_penalty 1.0,
retry przy ucięciu, historia w obu fazach, archiwum sesji JSON.

Uruchomienie / usage:
    # backend Unsloth ładuje model na JEDNĄ kartę
    # Unsloth backend loads the model on ONE card
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
        python ania_inference_panel_v3.py --model models/Ania_11B_Research_Preview

Uwaga: backend Unsloth ładuje model na jedną kartę — tak samo działał
panel v1, który przepracował tysiące lekcji.
"""

import os
import sys

# ------------------------------------------------------------
# Wybór GPU MUSI nastąpić przed importem torch.
# Dlatego --gpus jest czytane ręcznie z sys.argv, zanim
# argparse i torch zostaną w ogóle załadowane.
# ------------------------------------------------------------

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def _preparse_gpus(argv: list[str]) -> None:
    for index, argument in enumerate(argv):
        if argument == "--gpus" and index + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[index + 1].replace(" ", "")
            return

        if argument.startswith("--gpus="):
            os.environ["CUDA_VISIBLE_DEVICES"] = argument.split("=", 1)[1].replace(" ", "")
            return


_preparse_gpus(sys.argv[1:])

# Unsloth musi być zaimportowany PRZED transformers.
from unsloth import FastLanguageModel

import gc
import json
import random
import argparse
import threading
import traceback
import uuid

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import gradio as gr


# ============================================================
# KONFIGURACJA
# ============================================================

DEFAULT_CONFIG = {
    "MODEL_PATH": "brainhome/Ania_11B_Research_Preview",

    "HOST": "127.0.0.1",
    "PORT": 7860,

    "SESSIONS_DIR": "rozmowy_ania",

    "MAX_SEQ_LENGTH": 65536,
    "SAFETY_MARGIN_TOKENS": 4096,

    # Model publikowany działa wyłącznie w BF16 — bez kwantyzacji.
    "DTYPE": torch.bfloat16,

    "SEED": 12345,

    # Prywatna notatka robocza.
    "NOTE_MAX_TOKENS": 32768,
    "NOTE_RETRY_MAX_TOKENS": 49152,
    "NOTE_RETRY_ON_TRUNCATED": True,
    "NOTE_DO_SAMPLE": True,
    "NOTE_TEMPERATURE": 0.7,
    "NOTE_TOP_P": 0.95,

    # Finalna odpowiedź.
    "ANSWER_MAX_TOKENS": 16384,
    "ANSWER_RETRY_MAX_TOKENS": 24576,
    "ANSWER_RETRY_ON_TRUNCATED": True,
    "ANSWER_DO_SAMPLE": True,
    "ANSWER_TEMPERATURE": 0.7,
    "ANSWER_TOP_P": 0.95,

    # Celowo bez kary za powtórzenia.
    "REPETITION_PENALTY": 1.0,

    # EOS przy zużyciu co najmniej 90% budżetu.
    "SOFT_CAPITULATION_THRESHOLD": 0.9,

    "ALLOW_TF32": True,
}


# ============================================================
# PROMPTY
# ============================================================
# UWAGA WERYFIKACYJNA: w kopii v1 znaczniki <think> i </think>
# zostały prawdopodobnie utracone przy przenoszeniu pliku
# (interpretacja jako HTML). Poniżej są przywrócone — porównaj
# z oryginałem na dysku przed publikacją.

BASE_IDENTITY = """
Jesteś Ania Bielik. Rozmawiasz z użytkownikiem w panelu czatu.
""".strip()


PRIVATE_NOTE_SYSTEM = f"""
{BASE_IDENTITY}

W tej fazie przygotuj prywatną notatkę roboczą przed odpowiedzią.

Zasady techniczne:
- nie zwracaj się jeszcze finalnie do użytkownika,
- przeanalizuj ostatnią wiadomość w kontekście wcześniejszej rozmowy,
- sprawdź własne rozumowanie i ważne szczegóły,
- możesz rozważyć różne sposoby odpowiedzi,
- nie używaj znaczników [MYŚL], [/MYŚL], <think> ani </think>,
- nie dodawaj nagłówka opisującego proces myślenia,
- pisz po polsku.

Notatka jest prywatnym brudnopisem i nie zostanie pokazana użytkownikowi.
""".strip()


FINAL_ANSWER_SYSTEM = f"""
{BASE_IDENTITY}

Napisz finalną odpowiedź dla użytkownika.

Zasady techniczne:
- wykorzystaj prywatną notatkę roboczą,
- nie ujawniaj ani nie cytuj prywatnej notatki,
- nie wspominaj, że otrzymałaś notatkę,
- nie opisuj mechanizmu dwóch faz,
- nie używaj znaczników [MYŚL], [/MYŚL], <think> ani </think>,
- odpowiedz naturalnie, samodzielnie i zgodnie z kontekstem rozmowy.
""".strip()


# ============================================================
# STAN GLOBALNY
# ============================================================

MODEL = None
TOKENIZER = None
CONFIG = dict(DEFAULT_CONFIG)

SESSIONS: dict[str, dict[str, Any]] = {}

GENERATION_LOCK = threading.Lock()
SESSION_LOCK = threading.RLock()


# ============================================================
# NARZĘDZIA SYSTEMOWE
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def cleanup_cuda(deep: bool = False) -> None:
    gc.collect()

    if deep and torch.cuda.is_available():
        torch.cuda.empty_cache()

    if deep:
        gc.collect()


def ensure_tokenizer(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.pad_token_id


def primary_eos_token_id(tokenizer) -> int | None:
    eos = getattr(tokenizer, "eos_token_id", None)

    if isinstance(eos, int):
        return int(eos)

    if isinstance(eos, (list, tuple)) and eos:
        return int(eos[0])

    pad = getattr(tokenizer, "pad_token_id", None)

    if pad is not None:
        return int(pad)

    return None


def all_eos_token_ids(tokenizer) -> list[int]:
    result: set[int] = set()

    eos = getattr(tokenizer, "eos_token_id", None)

    if isinstance(eos, int):
        result.add(int(eos))

    elif isinstance(eos, (list, tuple, set)):
        for token_id in eos:
            try:
                result.add(int(token_id))
            except Exception:
                pass

    return sorted(result)


def get_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        try:
            return next(model.parameters()).device
        except Exception:
            return torch.device("cuda")


def json_safe(value):
    if isinstance(value, torch.dtype):
        return str(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, set):
        return sorted(value)

    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def log_generation(phase: str, result: dict[str, Any]) -> None:
    """Krótki log w terminalu po każdej generacji."""
    parts = [
        f"[{phase}]",
        f"input={result.get('input_tokens')}",
        f"generated={result.get('generated_tokens')}",
        f"finish={result.get('finish_reason')}",
    ]

    if result.get("was_retry"):
        parts.append("RETRY")

    if result.get("truncated"):
        parts.append("TRUNC")

    if result.get("soft_capitulation"):
        parts.append("SOFTCAP")

    if not result.get("ok"):
        parts.append(f"ERROR={result.get('error')}")

    print("  " + " | ".join(str(p) for p in parts))


class TokenProgressStreamer:
    """
    Licznik postępu generacji wypisywany do terminala.
    Nie pokazuje treści — tylko liczbę wygenerowanych tokenów,
    żeby było widać, że model żyje i jak daleko zaszedł.
    """

    def __init__(self, label: str, report_every: int = 1024):
        self.label = str(label)
        self.report_every = max(1, int(report_every))
        self.generated = 0
        self.prompt_consumed = False
        self.next_report = self.report_every

    def put(self, value):
        # Pierwsze wywołanie put() zawiera prompt — pomijamy je.
        if not self.prompt_consumed:
            self.prompt_consumed = True
            return

        try:
            count = int(value.numel())
        except Exception:
            count = 1

        self.generated += count

        if self.generated >= self.next_report:
            print(
                f"    [{self.label}] ~{self.generated} tokenów…",
                flush=True,
            )
            self.next_report += self.report_every

    def end(self):
        pass  # Podsumowanie wypisuje log_generation.


# ============================================================
# SESJE I ARCHIWUM
# ============================================================

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def create_session() -> dict[str, Any]:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]

    session = {
        "session_id": session_id,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "model": CONFIG["MODEL_PATH"],
        "backend": "unsloth",
        "turns": [],
    }

    with SESSION_LOCK:
        SESSIONS[session_id] = session

    save_session(session)
    return session


def get_or_create_session(session_id: str | None) -> dict[str, Any]:
    with SESSION_LOCK:
        if session_id and session_id in SESSIONS:
            return SESSIONS[session_id]

    return create_session()


def session_path(session: dict[str, Any]) -> Path:
    root = Path(CONFIG["SESSIONS_DIR"])
    root.mkdir(parents=True, exist_ok=True)

    return root / f"{session['session_id']}.json"


def save_session(session: dict[str, Any]) -> None:
    session["updated_at"] = now_iso()

    path = session_path(session)
    temporary_path = path.with_suffix(".json.tmp")

    payload = json.dumps(
        session,
        ensure_ascii=False,
        indent=2,
        default=json_safe,
    )

    temporary_path.write_text(payload, encoding="utf-8")
    temporary_path.replace(path)


def format_thinking_block(note_text: str) -> str:
    """Formatuje notatkę roboczą do wyświetlenia w oknie czatu."""
    return (
        "<details><summary>🧠 Notatka robocza (myślenie)</summary>\n\n"
        f"{note_text}\n\n"
        "</details>"
    )


def visible_history(
    session: dict[str, Any],
    show_thinking: bool = False,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    for turn in session.get("turns", []):
        messages.append(
            {
                "role": "user",
                "content": turn.get("user", ""),
            }
        )

        if show_thinking:
            note_text = (turn.get("reasoning_note") or {}).get("text", "")

            if note_text:
                messages.append(
                    {
                        "role": "assistant",
                        "content": format_thinking_block(note_text),
                    }
                )

        messages.append(
            {
                "role": "assistant",
                "content": turn.get("assistant", ""),
            }
        )

    return messages


def turns_to_messages(turns: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    for turn in turns:
        messages.append(
            {
                "role": "user",
                "content": str(turn.get("user", "")),
            }
        )

        messages.append(
            {
                "role": "assistant",
                "content": str(turn.get("assistant", "")),
            }
        )

    return messages


# ============================================================
# CZYSZCZENIE TEKSTU
# ============================================================

def clean_special_tokens(text: str) -> str:
    if not text:
        return ""

    result = str(text)

    unwanted = [
        "<|im_end|>",
        "<|im_start|>",
        "<s>",
        "</s>",
    ]

    for token in unwanted:
        result = result.replace(token, "")

    while "\n\n\n\n" in result:
        result = result.replace("\n\n\n\n", "\n\n\n")

    return result.strip()


def strip_thinking_markers(text: str) -> str:
    result = clean_special_tokens(text)

    markers = [
        "[MYŚL]",
        "[/MYŚL]",
        "[MYSL]",
        "[/MYSL]",
        "<think>",
        "</think>",
        "<thinking>",
        "</thinking>",
    ]

    for marker in markers:
        result = result.replace(marker, "")

    return result.strip()


def clean_private_note(text: str) -> str:
    return strip_thinking_markers(text)


def clean_final_answer(text: str) -> str:
    return strip_thinking_markers(text)


# ============================================================
# TOKENIZACJA I BUDŻET KONTEKSTU
# ============================================================

def tokenize_messages(messages: list[dict[str, str]]):
    """
    Czysty Transformers: return_dict=True i przeniesienie wszystkich
    tensorów na urządzenie embeddings (spójnie ze skryptem testowym MoE).
    """
    device = get_input_device(MODEL)

    inputs = TOKENIZER.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }

    return inputs


def count_message_tokens(messages: list[dict[str, str]]) -> int:
    input_ids = TOKENIZER.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    count = int(input_ids.shape[-1])
    del input_ids

    return count


def fit_history_to_budget(
    turns: list[dict[str, Any]],
    message_builder,
    requested_output_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    fitted_turns = deepcopy(turns)

    maximum_input_tokens = (
        int(CONFIG["MAX_SEQ_LENGTH"])
        - int(CONFIG["SAFETY_MARGIN_TOKENS"])
        - int(requested_output_tokens)
    )

    maximum_input_tokens = max(1, maximum_input_tokens)

    while True:
        messages = message_builder(fitted_turns)
        input_tokens = count_message_tokens(messages)

        if input_tokens <= maximum_input_tokens:
            return fitted_turns, input_tokens

        if not fitted_turns:
            return fitted_turns, input_tokens

        fitted_turns.pop(0)


def compute_effective_max_new_tokens(
    input_tokens: int,
    requested_max_new_tokens: int,
) -> int:
    remaining = (
        int(CONFIG["MAX_SEQ_LENGTH"])
        - int(input_tokens)
        - int(CONFIG["SAFETY_MARGIN_TOKENS"])
    )

    if remaining <= 0:
        return 0

    return max(1, min(int(requested_max_new_tokens), remaining))


# ============================================================
# GENERACJA
# ============================================================

def generate_text(
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    progress_label: str = "GENERACJA",
) -> dict[str, Any]:
    set_seed(seed)

    inputs = tokenize_messages(messages)
    input_ids = inputs["input_ids"]
    input_tokens = int(input_ids.shape[-1])

    effective_max = compute_effective_max_new_tokens(
        input_tokens=input_tokens,
        requested_max_new_tokens=max_new_tokens,
    )

    if effective_max <= 0:
        del inputs

        return {
            "ok": False,
            "text": "",
            "raw_text": "",
            "input_tokens": input_tokens,
            "effective_max_new_tokens": 0,
            "generated_tokens": 0,
            "ended_with_eos": False,
            "hit_token_limit": False,
            "truncated": True,
            "soft_capitulation": False,
            "finish_reason": "no_context_budget",
            "error": "Brak miejsca w kontekście na generację.",
        }

    primary_eos = primary_eos_token_id(TOKENIZER)
    eos_ids = all_eos_token_ids(TOKENIZER)

    eos_for_generate: int | list[int] | None

    if eos_ids:
        eos_for_generate = eos_ids
    else:
        eos_for_generate = primary_eos

    print(
        f"    [{progress_label}] start: input={input_tokens} tokenów, "
        f"budżet={effective_max} tokenów",
        flush=True,
    )

    generation_kwargs = {
        **inputs,
        "max_new_tokens": effective_max,
        "do_sample": bool(do_sample),
        "use_cache": True,
        "repetition_penalty": float(repetition_penalty),
        "pad_token_id": primary_eos,
        "eos_token_id": eos_for_generate,
        "streamer": TokenProgressStreamer(progress_label),
    }

    if do_sample:
        generation_kwargs["temperature"] = float(temperature)
        generation_kwargs["top_p"] = float(top_p)

    try:
        with torch.inference_mode():
            output = MODEL.generate(**generation_kwargs)

        generated_ids = output[0, input_ids.shape[-1]:]
        generated_tokens = int(generated_ids.numel())

        ended_with_eos = False

        if generated_tokens > 0 and eos_ids:
            last_token = int(generated_ids[-1].item())
            ended_with_eos = last_token in set(eos_ids)

        hit_token_limit = generated_tokens >= effective_max
        truncated = bool(hit_token_limit and not ended_with_eos)

        near_limit = generated_tokens >= int(
            float(CONFIG["SOFT_CAPITULATION_THRESHOLD"]) * effective_max
        )

        soft_capitulation = bool(ended_with_eos and near_limit)

        if ended_with_eos:
            finish_reason = "eos"
        elif truncated:
            finish_reason = "length"
        else:
            finish_reason = "stopped_or_unknown"

        raw_text = TOKENIZER.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        result = {
            "ok": True,
            "text": clean_special_tokens(raw_text),
            "raw_text": raw_text,
            "input_tokens": input_tokens,
            "effective_max_new_tokens": effective_max,
            "generated_tokens": generated_tokens,
            "ended_with_eos": ended_with_eos,
            "hit_token_limit": hit_token_limit,
            "truncated": truncated,
            "soft_capitulation": soft_capitulation,
            "finish_reason": finish_reason,
            "error": None,
        }

        del output
        del generated_ids
        del inputs

        return result

    except Exception as error:
        del inputs

        cleanup_cuda(deep=True)

        return {
            "ok": False,
            "text": "",
            "raw_text": "",
            "input_tokens": input_tokens,
            "effective_max_new_tokens": effective_max,
            "generated_tokens": 0,
            "ended_with_eos": False,
            "hit_token_limit": False,
            "truncated": False,
            "soft_capitulation": False,
            "finish_reason": "error",
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


def retry_if_truncated(
    first_result: dict[str, Any],
    *,
    retry_enabled: bool,
    retry_max_tokens: int,
    generation_function,
) -> dict[str, Any]:
    if not first_result.get("ok"):
        return first_result

    if not first_result.get("truncated"):
        return first_result

    if not retry_enabled:
        return first_result

    retry_result = generation_function(retry_max_tokens)

    if not retry_result.get("ok"):
        first_result["retry_failed"] = True
        first_result["retry_error"] = retry_result.get("error")
        return first_result

    retry_result["was_retry"] = True
    retry_result["first_attempt_truncated"] = True
    retry_result["first_attempt_generated_tokens"] = first_result.get(
        "generated_tokens"
    )
    retry_result["first_attempt_effective_max_new_tokens"] = first_result.get(
        "effective_max_new_tokens"
    )

    return retry_result


# ============================================================
# BUDOWANIE OBU FAZ
# ============================================================

def build_private_note_messages(
    previous_turns: list[dict[str, Any]],
    user_message: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": PRIVATE_NOTE_SYSTEM,
        },
        *turns_to_messages(previous_turns),
        {
            "role": "user",
            "content": user_message,
        },
    ]


def build_final_messages(
    previous_turns: list[dict[str, Any]],
    user_message: str,
    private_note: str,
) -> list[dict[str, str]]:
    final_user_message = (
        "[PRYWATNA NOTATKA ROBOCZA — NIE UJAWNIAĆ]\n"
        f"{private_note.strip()}\n"
        "[KONIEC PRYWATNEJ NOTATKI]\n\n"
        "[AKTUALNA WIADOMOŚĆ UŻYTKOWNIKA]\n"
        f"{user_message.strip()}"
    )

    return [
        {
            "role": "system",
            "content": FINAL_ANSWER_SYSTEM,
        },
        *turns_to_messages(previous_turns),
        {
            "role": "user",
            "content": final_user_message,
        },
    ]


def generate_private_note(
    previous_turns: list[dict[str, Any]],
    user_message: str,
    seed: int,
    temperature: float | None = None,
) -> tuple[dict[str, Any], int]:
    fitted_turns, dropped_count = fit_turns_for_note(
        previous_turns,
        user_message,
    )

    effective_temperature = (
        float(temperature)
        if temperature is not None
        else float(CONFIG["NOTE_TEMPERATURE"])
    )

    def run(max_tokens: int):
        messages = build_private_note_messages(
            fitted_turns,
            user_message,
        )

        result = generate_text(
            messages,
            max_new_tokens=max_tokens,
            do_sample=CONFIG["NOTE_DO_SAMPLE"],
            temperature=effective_temperature,
            top_p=CONFIG["NOTE_TOP_P"],
            repetition_penalty=CONFIG["REPETITION_PENALTY"],
            seed=seed,
            progress_label="NOTATKA",
        )

        result["text"] = clean_private_note(result.get("text", ""))
        return result

    first_result = run(CONFIG["NOTE_MAX_TOKENS"])

    result = retry_if_truncated(
        first_result,
        retry_enabled=CONFIG["NOTE_RETRY_ON_TRUNCATED"],
        retry_max_tokens=CONFIG["NOTE_RETRY_MAX_TOKENS"],
        generation_function=run,
    )

    log_generation("NOTATKA", result)

    return result, dropped_count


def fit_turns_for_note(
    previous_turns: list[dict[str, Any]],
    user_message: str,
) -> tuple[list[dict[str, Any]], int]:
    fitted, _ = fit_history_to_budget(
        previous_turns,
        message_builder=lambda turns: build_private_note_messages(
            turns,
            user_message,
        ),
        requested_output_tokens=CONFIG["NOTE_MAX_TOKENS"],
    )

    dropped = len(previous_turns) - len(fitted)
    return fitted, dropped


def fit_turns_for_final(
    previous_turns: list[dict[str, Any]],
    user_message: str,
    private_note: str,
) -> tuple[list[dict[str, Any]], int]:
    fitted, _ = fit_history_to_budget(
        previous_turns,
        message_builder=lambda turns: build_final_messages(
            turns,
            user_message,
            private_note,
        ),
        requested_output_tokens=CONFIG["ANSWER_MAX_TOKENS"],
    )

    dropped = len(previous_turns) - len(fitted)
    return fitted, dropped


def generate_final_answer(
    previous_turns: list[dict[str, Any]],
    user_message: str,
    private_note: str,
    seed: int,
    temperature: float | None = None,
) -> tuple[dict[str, Any], int]:
    fitted_turns, dropped_count = fit_turns_for_final(
        previous_turns,
        user_message,
        private_note,
    )

    effective_temperature = (
        float(temperature)
        if temperature is not None
        else float(CONFIG["ANSWER_TEMPERATURE"])
    )

    def run(max_tokens: int):
        messages = build_final_messages(
            fitted_turns,
            user_message,
            private_note,
        )

        result = generate_text(
            messages,
            max_new_tokens=max_tokens,
            do_sample=CONFIG["ANSWER_DO_SAMPLE"],
            temperature=effective_temperature,
            top_p=CONFIG["ANSWER_TOP_P"],
            repetition_penalty=CONFIG["REPETITION_PENALTY"],
            seed=seed,
            progress_label="ODPOWIEDŹ",
        )

        result["text"] = clean_final_answer(result.get("text", ""))
        return result

    first_result = run(CONFIG["ANSWER_MAX_TOKENS"])

    result = retry_if_truncated(
        first_result,
        retry_enabled=CONFIG["ANSWER_RETRY_ON_TRUNCATED"],
        retry_max_tokens=CONFIG["ANSWER_RETRY_MAX_TOKENS"],
        generation_function=run,
    )

    log_generation("ODPOWIEDŹ", result)

    return result, dropped_count


# ============================================================
# OBSŁUGA TURY
# ============================================================

def process_message(
    user_message: str,
    session_id: str | None,
    show_thinking: bool,
    note_temperature: float,
    answer_temperature: float,
):
    user_message = (user_message or "").strip()
    session = get_or_create_session(session_id)
    show_thinking = bool(show_thinking)

    if not user_message:
        yield (
            visible_history(session, show_thinking),
            session["session_id"],
            "",
            "Wpisz wiadomość.",
        )
        return

    current_visible = visible_history(session, show_thinking)
    current_visible.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    yield (
        current_visible,
        session["session_id"],
        "",
        "Faza 1/2 — Ania przygotowuje notatkę roboczą…",
    )

    with GENERATION_LOCK:
        try:
            previous_turns = deepcopy(session.get("turns", []))
            turn_number = len(previous_turns) + 1

            print(f"\n=== TURA {turn_number} | sesja {session['session_id']} ===")

            base_seed = (
                int(CONFIG["SEED"])
                + turn_number * 1000
                + sum(ord(character) for character in session["session_id"])
            )

            note_result, note_dropped_turns = generate_private_note(
                previous_turns=previous_turns,
                user_message=user_message,
                seed=base_seed,
                temperature=float(note_temperature),
            )

            if not note_result.get("ok"):
                raise RuntimeError(
                    "Nie udało się utworzyć prywatnej notatki: "
                    f"{note_result.get('error')}"
                )

            yield (
                current_visible,
                session["session_id"],
                "",
                "Faza 2/2 — Ania pisze odpowiedź…",
            )

            final_result, final_dropped_turns = generate_final_answer(
                previous_turns=previous_turns,
                user_message=user_message,
                private_note=note_result.get("text", ""),
                seed=base_seed + 1,
                temperature=float(answer_temperature),
            )

            if not final_result.get("ok"):
                raise RuntimeError(
                    "Nie udało się utworzyć odpowiedzi: "
                    f"{final_result.get('error')}"
                )

            assistant_text = final_result.get("text", "")

            turn = {
                "turn": turn_number,
                "created_at": now_iso(),
                "user": user_message,
                "assistant": assistant_text,

                "reasoning_note": {
                    "visible": False,
                    "text": note_result.get("text", ""),
                    "raw_text": note_result.get("raw_text", ""),
                    "metadata": {
                        key: value
                        for key, value in note_result.items()
                        if key not in {"text", "raw_text", "traceback"}
                    },
                },

                "generation": {
                    "answer": {
                        key: value
                        for key, value in final_result.items()
                        if key not in {"text", "raw_text", "traceback"}
                    },
                    "note_temperature": float(note_temperature),
                    "answer_temperature": float(answer_temperature),
                    "history_turns_dropped_for_note": note_dropped_turns,
                    "history_turns_dropped_for_answer": final_dropped_turns,
                },
            }

            with SESSION_LOCK:
                session["turns"].append(turn)
                save_session(session)

            cleanup_cuda(deep=False)

            yield (
                visible_history(session, show_thinking),
                session["session_id"],
                "",
                "Gotowe.",
            )

        except Exception as error:
            cleanup_cuda(deep=True)

            error_message = (
                "Wystąpił błąd podczas generacji. "
                f"Szczegóły zapisano w terminalu.\n\n`{error}`"
            )

            print("=" * 80)
            print("BŁĄD GENERACJI")
            print(traceback.format_exc())
            print("=" * 80)

            failed_visible = visible_history(session, show_thinking)
            failed_visible.append(
                {
                    "role": "user",
                    "content": user_message,
                }
            )
            failed_visible.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )

            yield (
                failed_visible,
                session["session_id"],
                user_message,
                "Błąd generacji.",
            )


def new_conversation():
    session = create_session()

    return (
        [],
        session["session_id"],
        "",
        "Nowa rozmowa.",
    )


def rerender_history(session_id: str | None, show_thinking: bool):
    """Przerysowuje okno czatu po przełączeniu widoczności myślenia."""
    with SESSION_LOCK:
        session = SESSIONS.get(session_id) if session_id else None

    if not session:
        return []

    return visible_history(session, bool(show_thinking))


# ============================================================
# INTERFEJS
# ============================================================

INTERFACE_CSS = """
#ania-title {
    text-align: center;
    margin-bottom: 0.2rem;
}

#ania-subtitle {
    text-align: center;
    opacity: 0.75;
    margin-bottom: 1rem;
}

#ania-status {
    min-height: 2.5rem;
}
"""


def build_interface():
    with gr.Blocks(
        title="Ania 11B Research Preview",
    ) as demo:
        gr.Markdown(
            "# Ania 11B Research Preview",
            elem_id="ania-title",
        )

        gr.Markdown(
            "Lokalny panel rozmowy",
            elem_id="ania-subtitle",
        )

        session_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1, min_width=230):
                gr.Markdown("### Ustawienia")

                show_thinking = gr.Checkbox(
                    label="Pokaż myślenie",
                    value=False,
                    info="Wyświetla notatkę roboczą nad każdą odpowiedzią.",
                )

                note_temperature = gr.Slider(
                    minimum=0.05,
                    maximum=1.5,
                    value=DEFAULT_CONFIG["NOTE_TEMPERATURE"],
                    step=0.05,
                    label="Temperatura — notatka",
                )

                answer_temperature = gr.Slider(
                    minimum=0.05,
                    maximum=1.5,
                    value=DEFAULT_CONFIG["ANSWER_TEMPERATURE"],
                    step=0.05,
                    label="Temperatura — odpowiedź",
                )

                new_button = gr.Button(
                    "Nowa rozmowa",
                    variant="secondary",
                )

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    height=650,
                    label="Rozmowa",
                    buttons=["copy"],
                )

                status = gr.Markdown(
                    "Model gotowy.",
                    elem_id="ania-status",
                )

                with gr.Row():
                    user_input = gr.Textbox(
                        label="Wiadomość",
                        placeholder="Napisz wiadomość do Ani…",
                        lines=3,
                        max_lines=12,
                        autofocus=True,
                        scale=8,
                    )

                    send_button = gr.Button(
                        "Wyślij",
                        variant="primary",
                        scale=1,
                    )

        event_inputs = [
            user_input,
            session_state,
            show_thinking,
            note_temperature,
            answer_temperature,
        ]

        event_outputs = [
            chatbot,
            session_state,
            user_input,
            status,
        ]

        send_button.click(
            fn=process_message,
            inputs=event_inputs,
            outputs=event_outputs,
            concurrency_limit=1,
        )

        user_input.submit(
            fn=process_message,
            inputs=event_inputs,
            outputs=event_outputs,
            concurrency_limit=1,
        )

        new_button.click(
            fn=new_conversation,
            inputs=None,
            outputs=event_outputs,
            concurrency_limit=1,
        )

        show_thinking.change(
            fn=rerender_history,
            inputs=[session_state, show_thinking],
            outputs=[chatbot],
            concurrency_limit=1,
        )

    return demo


# ============================================================
# URUCHOMIENIE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Panel inferencyjny dla Ania 11B Research Preview (v2, czysty Transformers)"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_CONFIG["MODEL_PATH"],
        help="Ścieżka lokalna lub identyfikator modelu na Hugging Face.",
    )

    parser.add_argument(
        "--gpus",
        default=None,
        help=(
            "Indeks karty GPU do użycia, np. '3'. Unsloth ładuje model "
            "na JEDNĄ kartę — ta flaga wybiera którą."
        ),
    )

    parser.add_argument(
        "--device-map",
        default=None,
        help=(
            "Eksperymentalne: np. 'balanced' próbuje rozłożyć model na "
            "wiele kart. Bez gwarancji wsparcia przez Unsloth."
        ),
    )

    parser.add_argument(
        "--host",
        default=DEFAULT_CONFIG["HOST"],
    )

    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_CONFIG["PORT"],
    )

    parser.add_argument(
        "--sessions-dir",
        default=DEFAULT_CONFIG["SESSIONS_DIR"],
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_CONFIG["MAX_SEQ_LENGTH"],
        help=(
            "Maksymalny kontekst. Na pojedynczej karcie 24 GB "
            "zalecane np. 16384."
        ),
    )

    parser.add_argument(
        "--no-tf32",
        action="store_true",
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Włącza tymczasowy publiczny link Gradio.",
    )

    return parser.parse_args()


def load_model():
    global MODEL
    global TOKENIZER

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(wszystkie)")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print("=" * 80)
    print("ŁADOWANIE MODELU ANI (backend Unsloth)")
    print("=" * 80)
    print(f"Model: {CONFIG['MODEL_PATH']}")
    print(f"CUDA_VISIBLE_DEVICES: {visible}")
    print(f"Widoczne karty GPU: {gpu_count}")
    print(f"Max context: {CONFIG['MAX_SEQ_LENGTH']}")
    print(f"Precyzja: {CONFIG['DTYPE']} (bez kwantyzacji)")
    print("=" * 80)

    # Zgodnie ze środowiskiem referencyjnym testu AIME: device_map="auto"
    # rozkłada model na widoczne karty (tak ładował skrypt, który dał 10/15).
    load_kwargs = dict(
        model_name=CONFIG["MODEL_PATH"],
        max_seq_length=CONFIG["MAX_SEQ_LENGTH"],
        dtype=CONFIG["DTYPE"],
        load_in_4bit=False,
        device_map=CONFIG.get("DEVICE_MAP") or "auto",
    )
    print(f"device_map: {load_kwargs['device_map']}")

    MODEL, TOKENIZER = FastLanguageModel.from_pretrained(**load_kwargs)

    FastLanguageModel.for_inference(MODEL)
    ensure_tokenizer(TOKENIZER)

    loaded_dtype = next(MODEL.parameters()).dtype
    loaded_device = next(MODEL.parameters()).device
    print(f"Rzeczywisty dtype wag: {loaded_dtype}")
    print(f"Urządzenie wag: {loaded_device}")

    print("Model gotowy.")
    print("=" * 80)


def main():
    args = parse_args()

    CONFIG["MODEL_PATH"] = args.model
    CONFIG["HOST"] = args.host
    CONFIG["PORT"] = args.port
    CONFIG["SESSIONS_DIR"] = args.sessions_dir
    CONFIG["MAX_SEQ_LENGTH"] = args.max_seq_length
    CONFIG["DEVICE_MAP"] = args.device_map

    if args.no_tf32:
        CONFIG["ALLOW_TF32"] = False

    torch.backends.cuda.matmul.allow_tf32 = bool(CONFIG["ALLOW_TF32"])
    torch.backends.cudnn.allow_tf32 = bool(CONFIG["ALLOW_TF32"])

    set_seed(CONFIG["SEED"])

    Path(CONFIG["SESSIONS_DIR"]).mkdir(
        parents=True,
        exist_ok=True,
    )

    load_model()

    demo = build_interface()

    demo.queue(
        default_concurrency_limit=1,
        max_size=8,
    )

    demo.launch(
        server_name=CONFIG["HOST"],
        server_port=CONFIG["PORT"],
        share=args.share,
        show_error=True,
        css=INTERFACE_CSS,
    )


if __name__ == "__main__":
    main()
