# ========================================
# aime_eval_ania.py
# ========================================
"""
AIME 2026 — referencyjny skrypt ewaluacyjny dla Ania 11B Research Preview.
Reference evaluation script for Ania 11B Research Preview.

To jest DOKŁADNIE to środowisko (tryb dwufazowy: prywatna notatka robocza ->
finalna odpowiedź; parametry, budżety tokenów, retry, detekcja soft
capitulation), w którym uzyskano wynik 10/15 raportowany w karcie modelu.
This is the exact two-phase environment (private working note -> final
answer) used to obtain the 10/15 AIME 2026 result reported in the model card.

Parametry referencyjne / reference sampling:
    temperature 0.7, top_p 0.95, repetition_penalty 1.0 (bez kary!),
    BF16, max context 65536.
Nie używaj greedy decodingu ani kary za powtórzenia — oba mierzalnie
degradują rozumowanie modelu. / Do not switch to greedy decoding and do
not add a repetition penalty — both measurably degrade reasoning.

WAŻNE — zadania AIME NIE są dołączone (prawa autorskie MAA).
IMPORTANT — the AIME problems are NOT distributed here (MAA copyright).
Przygotuj własny plik JSONL: jedna linia = jedno zadanie,
    {"question": "...", "answer": 123}
w kolejności oficjalnych zadań (id = numer linii) i wskaż flagą --aime.
Provide your own JSONL (one problem per line, official order; id = line
number) and pass it via --aime.

Użycie / usage:
    # jedna karta / one card
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
        python aime_eval_ania.py --model brainhome/Ania_11B_Research_Preview --aime aime2026.jsonl

    # dwie karty, para z mostkiem NVLink / two cards, bridged NVLink pair
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3 \
        python aime_eval_ania.py --model brainhome/Ania_11B_Research_Preview --aime aime2026.jsonl
"""

import os

# WAŻNE: przed importem torch / unsloth.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import sys


def _preparse_gpus(argv):
    """Wybór GPU MUSI nastąpić przed importem torch — dlatego --gpus jest
    czytane ręcznie z sys.argv, zanim argparse i torch zostaną załadowane.
    GPU selection MUST happen before torch is imported."""
    for index, argument in enumerate(argv):
        if argument == "--gpus" and index + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[index + 1].replace(" ", "")
            return
        if argument.startswith("--gpus="):
            os.environ["CUDA_VISIBLE_DEVICES"] = argument.split("=", 1)[1].replace(" ", "")
            return


_preparse_gpus(sys.argv[1:])

import gc
import re
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm

from unsloth import FastLanguageModel


# ============================================================
# KONFIGURACJA DOMYŚLNA
# ============================================================

DEFAULT_CONFIG = {
    "MODEL_PATH": "models/Ania_11B_Research_Preview",
    "AIME_FILE": "test/aime2026.jsonl",

    "OUTPUT_ROOT": "raporty_testowe",

    "SEED": 12345,
    "MAX_SEQ_LENGTH": 65536,
    "SAFETY_MARGIN_TOKENS": 4096,
    "DTYPE": torch.bfloat16,

    # AIME — prywatna analiza + finalna odpowiedź.
    # v4: notatki dostają więcej przestrzeni (Q9/Q13 dusiły się przy 24k),
    # odpowiedzi mniej (realne odpowiedzi to 0.5-2k tokenów).
    "AIME_NOTE_MAX_TOKENS": 32768,
    "AIME_NOTE_RETRY_MAX_TOKENS": 49152,
    "AIME_NOTE_RETRY_ON_TRUNCATED": True,
    "AIME_NOTE_DO_SAMPLE": True,
    "AIME_NOTE_TEMPERATURE": 0.7,
    "AIME_NOTE_TOP_P": 0.95,

    "AIME_ANSWER_MAX_TOKENS": 8192,
    "AIME_ANSWER_RETRY_MAX_TOKENS": 16384,
    "AIME_ANSWER_RETRY_ON_TRUNCATED": True,
    "AIME_ANSWER_DO_SAMPLE": True,
    "AIME_ANSWER_TEMPERATURE": 0.7,
    "AIME_ANSWER_TOP_P": 0.95,

    # AIME: bez kary za powtórzenia.
    # Przy matematyce repeated symbols/cyfry/struktury są naturalne.
    "AIME_REPETITION_PENALTY": 1.0,

    # Przy uciętej odpowiedzi nie łapiemy ostatniej przypadkowej liczby z tekstu.
    "ALLOW_FALLBACK_EXTRACT_IF_TRUNCATED": False,

    # v4: próg miękkiej kapitulacji — EOS przy >= tym ułamku zużytego budżetu
    # oznaczamy flagą soft_capitulation (model "zamknął na skróty" pod presją).
    "SOFT_CAPITULATION_THRESHOLD": 0.9,

    # Co ile pytań robić cięższe sprzątanie CUDA.
    "CLEANUP_EVERY": 5,

    # Jeśli plik wyniku istnieje, pomijamy pytanie.
    "RESUME": True,

    # TF32 przyspiesza, ale nie jest idealnie deterministyczne.
    # Jeśli chcesz maksymalną powtarzalność AIME, ustaw False.
    "ALLOW_TF32": True,
}


# ============================================================
# PROMPTY
# ============================================================

AIME_PRIVATE_NOTE_SYSTEM = """
Jesteś Ania Bielik. Rozwiązujesz zadanie matematyczne w trybie prywatnej notatki roboczej.

To jest etap wewnętrzny:
- nie pisz finalnej odpowiedzi dla użytkownika,
- nie używaj znaczników [MYŚL], [/MYŚL], <think>, </think>,
- nie udawaj formalnego raportu,
- zapisz rozumowanie robocze po polsku,
- sprawdzaj rachunki i warunki zadania,
- jeśli są możliwe pułapki, nazwij je w notatce,
- na końcu prywatnej notatki zapisz roboczy kandydat na wynik.

Ta notatka zostanie użyta w drugiej fazie do wygenerowania finalnej odpowiedzi.
""".strip()


AIME_FINAL_SYSTEM = """
Jesteś Ania Bielik. Odpowiadasz na zadanie AIME.

Zasady:
- odpowiedzi AIME są liczbami całkowitymi od 0 do 999, jeśli wychodzi ci wynik poza tym zakresem zrób zadanie ponownie bo wynik jest nieprawidłowy!
- wykorzystaj prywatną notatkę roboczą,
- nie ujawniaj, że dostałaś prywatną notatkę,
- nie cytuj notatki jako osobnego artefaktu,
- odpowiedz zwięźle,
- na końcu bezwzględnie podaj wynik w formacie \\boxed{liczba}.
""".strip()


# ============================================================
# SYSTEM / GPU
# ============================================================

def set_seed(seed: int):
    random.seed(int(seed))
    torch.manual_seed(int(seed))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def light_cleanup():
    gc.collect()


def deep_cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()


def get_output_dir_name(model_path: str, output_root: str) -> Path:
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).name.rstrip("/") or "model"
    out = Path(output_root) / f"raporty_{date_str}_{model_name}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_input_device(model):
    """
    Przy device_map='auto' model może być pocięty po kartach.
    Input zwykle powinien trafić na urządzenie embeddingów.
    """
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        try:
            return next(model.parameters()).device
        except Exception:
            return torch.device("cuda")


def ensure_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.eos_token_id is None and tokenizer.pad_token_id is not None:
        tokenizer.eos_token_id = tokenizer.pad_token_id


def primary_eos_token_id(tokenizer):
    eos = getattr(tokenizer, "eos_token_id", None)

    if isinstance(eos, int):
        return eos

    if isinstance(eos, (list, tuple)) and eos:
        return int(eos[0])

    return getattr(tokenizer, "pad_token_id", None)


def eos_token_ids(tokenizer) -> set[int]:
    ids = set()
    eos = getattr(tokenizer, "eos_token_id", None)

    if isinstance(eos, int):
        ids.add(int(eos))
    elif isinstance(eos, (list, tuple, set)):
        for x in eos:
            try:
                ids.add(int(x))
            except Exception:
                pass

    return ids


def safe_int_for_seed(value, fallback: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        s = str(value)
        n = 0
        for ch in s:
            n = (n * 131 + ord(ch)) % 1000000
        return n or fallback


# ============================================================
# ŁADOWANIE DANYCH
# ============================================================

def load_aime_dataset(path: str | Path) -> list[dict]:
    path = Path(path)
    suite = []

    if not path.exists():
        print(f"⚠️ Brak pliku AIME: {path}")
        return suite

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception:
                continue

            # Obsługiwane klucze pytania / supported question keys:
            #   question | problem
            # Obsługiwane klucze odpowiedzi / supported answer keys:
            #   answer | final_answer | ground_truth
            q = item.get("question")
            if q is None:
                q = item.get("problem")

            a = item.get("answer")
            if a is None:
                a = item.get("final_answer")
            if a is None:
                a = item.get("ground_truth")

            if q and a is not None:
                suite.append(
                    {
                        "id": item.get("id") or i,
                        "question": str(q).strip(),
                        "answer": str(a).strip(),
                    }
                )

    return suite


# ============================================================
# CZYSZCZENIE TEKSTU
# ============================================================

def clean_special_tokens(text: str) -> str:
    if not text:
        return ""

    t = str(text)

    unwanted = [
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
        "</s>",
    ]

    for u in unwanted:
        t = t.replace(u, "")

    t = re.sub(r"\n{4,}", "\n\n\n", t)
    return t.strip()


def strip_thinking_tags(text: str) -> str:
    """
    Czyści śmieci po tagach myślenia.
    Nie zakładamy, że model poprawnie zamknie tag.
    """
    if not text:
        return ""

    t = clean_special_tokens(text)

    replacements = [
        "[MYŚL]",
        "[/MYŚL]",
        "[MYSL]",
        "[/MYSL]",
        "[MYŚL]",
        "[/MYŚL]",
        "<think>",
        "</think>",
        "<thinking>",
        "</thinking>",
    ]

    for r in replacements:
        t = t.replace(r, "")

    t = re.sub(r"<think\b[^>]*>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"</think\s*>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<thinking\b[^>]*>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"</thinking\s*>", "", t, flags=re.IGNORECASE)

    # Usuwanie nagłówków, jeśli model je mimo wszystko doda.
    t = re.sub(
        r"(?im)^\s*(prywatna\s+)?(myśl|mysl|analiza|brudnopis|notatka\s+robocza)\s*:\s*$",
        "",
        t,
    )

    t = re.sub(r"\n{4,}", "\n\n\n", t)
    return t.strip()


def clean_private_note(text: str) -> str:
    t = strip_thinking_tags(text)

    # NIE ucinamy "Wynik:" ani "Odpowiedź:".
    # Prompt prosi model, żeby na końcu notatki zapisał roboczy kandydat na wynik.
    # To musi przeżyć do fazy finalnej.
    cut_patterns = [
        r"(?im)^\s*finalna odpowied[źz]\s*:",
        r"(?im)^\s*ostateczna odpowied[źz]\s*:",
        r"(?im)^\s*final answer\s*:",
    ]

    cut_positions = []

    for pat in cut_patterns:
        m = re.search(pat, t)
        if m:
            cut_positions.append(m.start())

    if cut_positions:
        t = t[: min(cut_positions)].strip()

    return t.strip()


def clean_final_answer(text: str) -> str:
    t = strip_thinking_tags(text)

    private_block_patterns = [
        r"\[PRYWATNA NOTATKA ROBOCZA.*?\[KONIEC NOTATKI\]",
        r"\[NOTATKA ROBOCZA.*?\[KONIEC NOTATKI\]",
    ]

    for pat in private_block_patterns:
        t = re.sub(pat, "", t, flags=re.DOTALL | re.IGNORECASE)

    t = re.sub(r"\n{4,}", "\n\n\n", t)
    return t.strip()


# ============================================================
# EKSTRAKCJA AIME
# ============================================================

def extract_answer(text: str, *, allow_fallback: bool = True) -> str | None:
    if not text:
        return None

    t = clean_special_tokens(text)

    patterns = [
        r"\\boxed\s*\{\s*([0-9]{1,3})\s*\}",
        r"(?:FINAL|ANSWER|WYNIK|ODPOWIEDŹ|ODPOWIEDZ)[:\s]+([0-9]{1,3})",
        r"(?:wynik wynosi|odpowied[źz] (?:to|wynosi|brzmi)|końcowa odpowied[źz])[:\s]+([0-9]{1,3})",
    ]

    # \boxed — pierwsze trafienie (format wymuszony promptem).
    m = re.search(patterns[0], t, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Frazy słowne: bierzemy OSTATNIE trafienie, nie pierwsze.
    # "Wynik: 5" w środku rachunku to wartość pośrednia — finalna pada na końcu.
    for pat in patterns[1:]:
        matches = list(re.finditer(pat, t, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip()

    # Fallback jest ryzykowny.
    # Przy uciętej odpowiedzi nie wolno go używać.
    if not allow_fallback:
        return None

    tail = t[-500:]
    nums = re.findall(r"\b([0-9]{1,3})\b", tail)

    if nums:
        return nums[-1]

    return None


def normalize_aime_answer(value: str | int | None) -> int | None:
    if value is None:
        return None

    s = str(value).strip()
    m = re.search(r"\d+", s)

    if not m:
        return None

    try:
        return int(m.group(0))
    except Exception:
        return None


def aime_answers_equal(got: str | int | None, expected: str | int | None) -> bool:
    got_n = normalize_aime_answer(got)
    expected_n = normalize_aime_answer(expected)

    if got_n is None or expected_n is None:
        return False

    return got_n == expected_n


# ============================================================
# GENERACJA
# ============================================================

def build_input(tokenizer, messages: list[dict], device):
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    attention_mask = torch.ones_like(input_ids, device=device)
    return input_ids, attention_mask


def compute_effective_max_new_tokens(
    input_ids,
    requested_max_new_tokens: int,
    max_seq_length: int,
    safety_margin_tokens: int,
) -> int:
    input_tokens = int(input_ids.shape[-1])
    budget = int(max_seq_length) - input_tokens - int(safety_margin_tokens)

    if budget <= 0:
        return 0

    return max(1, min(int(requested_max_new_tokens), budget))


def generate_text(
    model,
    tokenizer,
    messages: list[dict],
    *,
    max_new_tokens: int,
    max_seq_length: int,
    safety_margin_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    soft_capitulation_threshold: float = 0.9,
) -> dict:
    set_seed(seed)

    device = get_input_device(model)
    input_ids, attention_mask = build_input(tokenizer, messages, device)

    effective_max = compute_effective_max_new_tokens(
        input_ids=input_ids,
        requested_max_new_tokens=max_new_tokens,
        max_seq_length=max_seq_length,
        safety_margin_tokens=safety_margin_tokens,
    )

    input_tokens = int(input_ids.shape[-1])

    if effective_max <= 0:
        del input_ids, attention_mask
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
            "error": "Brak budżetu kontekstu na generację.",
        }

    eos_id = primary_eos_token_id(tokenizer)

    # v4: generacja zatrzymuje się na KAŻDYM wariancie EOS, nie tylko primary.
    # Bez tego alternatywne tokeny końca przelatują do limitu i fałszywie
    # raportują truncated.
    all_eos = sorted(eos_token_ids(tokenizer))
    eos_for_generate = all_eos if all_eos else eos_id

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": effective_max,
        "do_sample": bool(do_sample),
        "use_cache": True,
        "repetition_penalty": float(repetition_penalty),
        "pad_token_id": eos_id,
        "eos_token_id": eos_for_generate,
    }

    if do_sample:
        generation_kwargs["temperature"] = float(temperature)
        generation_kwargs["top_p"] = float(top_p)

    try:
        with torch.inference_mode():
            out = model.generate(**generation_kwargs)

        generated_ids = out[0][input_ids.shape[-1]:]
        generated_tokens = int(generated_ids.numel())

        eos_ids = eos_token_ids(tokenizer)

        if generated_tokens > 0 and eos_ids:
            last_token_id = int(generated_ids[-1].item())
            ended_with_eos = last_token_id in eos_ids
        else:
            ended_with_eos = False

        hit_token_limit = generated_tokens >= int(effective_max)

        # Najważniejsza flaga:
        # jeśli wygenerowaliśmy cały budżet i nie skończyliśmy EOS-em,
        # traktujemy tekst jako ucięty.
        truncated = bool(hit_token_limit and not ended_with_eos)

        # v4: miękka kapitulacja — model skończył EOS-em, ale zużył >=90%
        # budżetu. Sygnał "zamknięcia na skróty" pod presją limitu
        # (konfabulacja / generalizacja z małej próbki tuż pod sufitem).
        near_limit = generated_tokens >= int(
            float(soft_capitulation_threshold) * effective_max
        )
        soft_capitulation = bool(ended_with_eos and near_limit)

        if ended_with_eos:
            finish_reason = "eos"
        elif truncated:
            finish_reason = "length"
        else:
            finish_reason = "stopped_or_unknown"

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        del out, generated_ids, input_ids, attention_mask

        return {
            "ok": True,
            "text": clean_special_tokens(text),
            "raw_text": text,
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

    except Exception as e:
        del input_ids, attention_mask

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
            "error": str(e),
        }


# ============================================================
# DWUFAZOWY TRYB: NOTATKA -> ODPOWIEDŹ
# ============================================================

def generate_private_note(
    model,
    tokenizer,
    question: str,
    *,
    system_prompt: str,
    config: dict,
    seed: int,
    max_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> dict:
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"Zadanie / pytanie:\n{question}",
        },
    ]

    result = generate_text(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_tokens,
        max_seq_length=config["MAX_SEQ_LENGTH"],
        safety_margin_tokens=config["SAFETY_MARGIN_TOKENS"],
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        soft_capitulation_threshold=config.get("SOFT_CAPITULATION_THRESHOLD", 0.9),
    )

    # raw_text z generate_text zostaje nietknięty (pełny zrzut do diagnostyki ucięć).
    result["text"] = clean_private_note(result.get("text", ""))

    return result


def generate_final_answer(
    model,
    tokenizer,
    question: str,
    private_note: str,
    *,
    system_prompt: str,
    config: dict,
    seed: int,
    max_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    aime_mode: bool = False,
) -> dict:
    private_note_block = (
        "[PRYWATNA NOTATKA ROBOCZA — NIE UJAWNIAĆ]\n"
        f"{private_note.strip()}\n"
        "[KONIEC NOTATKI]\n\n"
        "Na podstawie rozmowy i prywatnej notatki odpowiedz użytkownikowi finalnie. "
        "Nie wspominaj o notatce. Nie pokazuj procesu analizy."
    )

    if aime_mode:
        user_content = (
            f"{private_note_block}\n\n"
            f"Zadanie AIME:\n{question}\n\n"
            "Podaj krótkie rozwiązanie i zakończ dokładnie formatem \\boxed{liczba}."
        )
    else:
        user_content = (
            f"{private_note_block}\n\n"
            f"Pytanie użytkownika:\n{question}"
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    result = generate_text(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_tokens,
        max_seq_length=config["MAX_SEQ_LENGTH"],
        safety_margin_tokens=config["SAFETY_MARGIN_TOKENS"],
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        soft_capitulation_threshold=config.get("SOFT_CAPITULATION_THRESHOLD", 0.9),
    )

    # raw_text z generate_text zostaje nietknięty.
    result["text"] = clean_final_answer(result.get("text", ""))

    return result


def maybe_retry_generation(
    *,
    label: str,
    first_result: dict,
    retry_enabled: bool,
    retry_fn,
):
    if not first_result.get("ok"):
        return first_result

    if not first_result.get("truncated"):
        return first_result

    if not retry_enabled:
        return first_result

    print(
        f"⚠️ {label}: generacja ucięta "
        f"(generated={first_result.get('generated_tokens')} / "
        f"limit={first_result.get('effective_max_new_tokens')}). "
        f"Ponawiam z większym budżetem..."
    )

    retry_result = retry_fn()

    # Jeśli ponowienie się nie powiodło (błąd/OOM przy większym budżecie),
    # zostajemy przy pierwszym podejściu — ucięta notatka jest lepsza niż żadna.
    if not retry_result.get("ok"):
        first_result["retry_failed"] = True
        first_result["retry_error"] = retry_result.get("error")
        return first_result

    retry_result["was_retry"] = True
    retry_result["first_attempt_truncated"] = True
    retry_result["first_attempt_generated_tokens"] = first_result.get("generated_tokens")
    retry_result["first_attempt_effective_max_new_tokens"] = first_result.get("effective_max_new_tokens")

    return retry_result


# ============================================================
# ZAPIS
# ============================================================

def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def json_safe(obj):
    if isinstance(obj, torch.dtype):
        return str(obj)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (set, tuple)):
        return list(obj)

    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=json_safe),
        encoding="utf-8",
    )


def metadata_block(note_result: dict, answer_result: dict) -> str:
    return (
        f"- note_ok: {note_result.get('ok')}\n"
        f"- note_input_tokens: {note_result.get('input_tokens')}\n"
        f"- note_effective_max_new_tokens: {note_result.get('effective_max_new_tokens')}\n"
        f"- note_generated_tokens: {note_result.get('generated_tokens')}\n"
        f"- note_truncated: {note_result.get('truncated')}\n"
        f"- note_soft_capitulation: {note_result.get('soft_capitulation', False)}\n"
        f"- note_finish_reason: {note_result.get('finish_reason')}\n"
        f"- note_was_retry: {note_result.get('was_retry', False)}\n"
        f"- note_first_attempt_truncated: {note_result.get('first_attempt_truncated', False)}\n"
        f"- answer_ok: {answer_result.get('ok')}\n"
        f"- answer_input_tokens: {answer_result.get('input_tokens')}\n"
        f"- answer_effective_max_new_tokens: {answer_result.get('effective_max_new_tokens')}\n"
        f"- answer_generated_tokens: {answer_result.get('generated_tokens')}\n"
        f"- answer_truncated: {answer_result.get('truncated')}\n"
        f"- answer_soft_capitulation: {answer_result.get('soft_capitulation', False)}\n"
        f"- answer_finish_reason: {answer_result.get('finish_reason')}\n"
        f"- answer_was_retry: {answer_result.get('was_retry', False)}\n"
        f"- answer_first_attempt_truncated: {answer_result.get('first_attempt_truncated', False)}\n"
        f"- note_error: {note_result.get('error')}\n"
        f"- answer_error: {answer_result.get('error')}\n"
    )


def save_aime_case(
    output_dir: Path,
    item: dict,
    note_result: dict,
    answer_result: dict,
    extracted: str | None,
    ok: bool,
):
    qid = item["id"]

    log_text = (
        f"# AIME Q{qid}\n\n"
        f"## Pytanie\n\n{item['question']}\n\n"
        f"## Poprawna odpowiedź\n\n{item['answer']}\n\n"
        f"## Wynik modelu\n\n{extracted}\n\n"
        f"## Status\n\n{'POPRAWNA' if ok else 'BŁĘDNA'}\n\n"
        f"## Prywatna notatka robocza\n\n{note_result.get('text', '')}\n\n"
        f"## Finalna odpowiedź\n\n{answer_result.get('text', '')}\n\n"
        f"## Metadata\n\n"
        f"{metadata_block(note_result, answer_result)}"
    )

    write_text(output_dir / f"AIME_Q{qid}.md", log_text)


# ============================================================
# FAZA AIME
# ============================================================

def run_aime(model, tokenizer, dataset: list[dict], output_dir: Path, config: dict) -> list[dict]:
    summary = []

    if not dataset:
        return summary

    summary_path = output_dir / "aime_results.jsonl"

    for idx, item in enumerate(tqdm(dataset, desc="AIME", unit="zad"), start=1):
        qid = item["id"]
        qid_seed = safe_int_for_seed(qid, fallback=idx)
        case_path = output_dir / f"AIME_Q{qid}.md"

        if config["RESUME"] and case_path.exists():
            tqdm.write(f"⏭️ AIME Q{qid}: istnieje, pomijam.")
            continue

        seed = config["SEED"] + qid_seed * 5000

        note_result_first = generate_private_note(
            model,
            tokenizer,
            item["question"],
            system_prompt=AIME_PRIVATE_NOTE_SYSTEM,
            config=config,
            seed=seed,
            max_tokens=config["AIME_NOTE_MAX_TOKENS"],
            do_sample=config["AIME_NOTE_DO_SAMPLE"],
            temperature=config["AIME_NOTE_TEMPERATURE"],
            top_p=config["AIME_NOTE_TOP_P"],
            repetition_penalty=config["AIME_REPETITION_PENALTY"],
        )

        note_result = maybe_retry_generation(
            label=f"AIME Q{qid} note",
            first_result=note_result_first,
            retry_enabled=bool(config.get("AIME_NOTE_RETRY_ON_TRUNCATED")),
            retry_fn=lambda: generate_private_note(
                model,
                tokenizer,
                item["question"],
                system_prompt=AIME_PRIVATE_NOTE_SYSTEM,
                config=config,
                seed=seed,
                max_tokens=config["AIME_NOTE_RETRY_MAX_TOKENS"],
                do_sample=config["AIME_NOTE_DO_SAMPLE"],
                temperature=config["AIME_NOTE_TEMPERATURE"],
                top_p=config["AIME_NOTE_TOP_P"],
                repetition_penalty=config["AIME_REPETITION_PENALTY"],
            ),
        )

        if not note_result.get("ok"):
            answer_result = {
                "ok": False,
                "text": "",
                "raw_text": "",
                "error": "Pominięto odpowiedź, bo notatka robocza nie powstała.",
                "truncated": False,
                "soft_capitulation": False,
                "finish_reason": "skipped",
                "generated_tokens": 0,
                "effective_max_new_tokens": 0,
                "input_tokens": None,
            }
            extracted = None
            ok = False

        else:
            answer_result_first = generate_final_answer(
                model,
                tokenizer,
                item["question"],
                note_result.get("text", ""),
                system_prompt=AIME_FINAL_SYSTEM,
                config=config,
                seed=seed + 1,
                max_tokens=config["AIME_ANSWER_MAX_TOKENS"],
                do_sample=config["AIME_ANSWER_DO_SAMPLE"],
                temperature=config["AIME_ANSWER_TEMPERATURE"],
                top_p=config["AIME_ANSWER_TOP_P"],
                repetition_penalty=config["AIME_REPETITION_PENALTY"],
                aime_mode=True,
            )

            answer_result = maybe_retry_generation(
                label=f"AIME Q{qid} answer",
                first_result=answer_result_first,
                retry_enabled=bool(config.get("AIME_ANSWER_RETRY_ON_TRUNCATED")),
                retry_fn=lambda: generate_final_answer(
                    model,
                    tokenizer,
                    item["question"],
                    note_result.get("text", ""),
                    system_prompt=AIME_FINAL_SYSTEM,
                    config=config,
                    seed=seed + 1,
                    max_tokens=config["AIME_ANSWER_RETRY_MAX_TOKENS"],
                    do_sample=config["AIME_ANSWER_DO_SAMPLE"],
                    temperature=config["AIME_ANSWER_TEMPERATURE"],
                    top_p=config["AIME_ANSWER_TOP_P"],
                    repetition_penalty=config["AIME_REPETITION_PENALTY"],
                    aime_mode=True,
                ),
            )

            # Jeśli odpowiedź jest ucięta, nie wolno fallbackiem łapać ostatniej liczby z ogona.
            allow_fallback = True

            if answer_result.get("truncated") and not config.get("ALLOW_FALLBACK_EXTRACT_IF_TRUNCATED", False):
                allow_fallback = False

            extracted = extract_answer(
                answer_result.get("text", ""),
                allow_fallback=allow_fallback,
            )

            ok = aime_answers_equal(extracted, item["answer"])

        save_aime_case(
            output_dir=output_dir,
            item=item,
            note_result=note_result,
            answer_result=answer_result,
            extracted=extracted,
            ok=ok,
        )

        row = {
            "id": qid,
            "is_correct": ok,
            "expected": item["answer"],
            "got": extracted,

            "note_ok": note_result.get("ok"),
            "answer_ok": answer_result.get("ok"),

            "note_truncated": note_result.get("truncated"),
            "answer_truncated": answer_result.get("truncated"),

            "note_soft_capitulation": note_result.get("soft_capitulation", False),
            "answer_soft_capitulation": answer_result.get("soft_capitulation", False),

            "note_finish_reason": note_result.get("finish_reason"),
            "answer_finish_reason": answer_result.get("finish_reason"),

            "note_generated_tokens": note_result.get("generated_tokens"),
            "answer_generated_tokens": answer_result.get("generated_tokens"),

            "note_effective_max_new_tokens": note_result.get("effective_max_new_tokens"),
            "answer_effective_max_new_tokens": answer_result.get("effective_max_new_tokens"),

            "note_was_retry": note_result.get("was_retry", False),
            "answer_was_retry": answer_result.get("was_retry", False),

            "note_error": note_result.get("error"),
            "answer_error": answer_result.get("error"),
        }

        append_jsonl(summary_path, row)
        summary.append(row)

        status = "✅" if ok else "❌"
        trunc = ""

        if note_result.get("truncated"):
            trunc += " | note=TRUNCATED"

        if answer_result.get("truncated"):
            trunc += " | answer=TRUNCATED"

        if note_result.get("soft_capitulation"):
            trunc += " | note=SOFTCAP"

        if answer_result.get("soft_capitulation"):
            trunc += " | answer=SOFTCAP"

        if note_result.get("was_retry"):
            trunc += " | note=RETRY"

        if answer_result.get("was_retry"):
            trunc += " | answer=RETRY"

        tqdm.write(
            f"AIME Q{qid}: {status} | model={extracted} | expected={item['answer']}{trunc}"
        )

        if idx % int(config["CLEANUP_EVERY"]) == 0:
            deep_cleanup()
        else:
            light_cleanup()

    return summary


# ============================================================
# RAPORT KOŃCOWY
# ============================================================

def build_final_report(
    output_dir: Path,
    config: dict,
    aime_summary: list[dict],
):
    report_path = output_dir / "RAPORT_KONCOWY.md"

    lines = []
    lines.append("# RAPORT Z TESTU MODELU")
    lines.append("")
    lines.append(f"- Data wykonania: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- Model: `{config['MODEL_PATH']}`")
    lines.append(f"- Max context: `{config['MAX_SEQ_LENGTH']}`")
    lines.append(f"- AIME cases: `{len(aime_summary)}`")
    lines.append(f"- AIME note budget: `{config['AIME_NOTE_MAX_TOKENS']}` (retry `{config['AIME_NOTE_RETRY_MAX_TOKENS']}`)")
    lines.append(f"- AIME repetition penalty: `{config['AIME_REPETITION_PENALTY']}`")
    lines.append(f"- Soft capitulation threshold: `{config['SOFT_CAPITULATION_THRESHOLD']}`")
    lines.append(f"- TF32: `{config['ALLOW_TF32']}`")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## AIME")
    lines.append("")

    if aime_summary:
        correct = sum(1 for r in aime_summary if r.get("is_correct"))
        total = len(aime_summary)
        acc = correct / total * 100 if total else 0.0

        lines.append(f"**Skuteczność:** {correct} / {total} ({acc:.2f}%)")
        lines.append("")
        lines.append("| ID | Status | Model | Oczekiwano | Note trunc | Note softcap | Answer trunc | Retry |")
        lines.append("|----|--------|-------|------------|------------|--------------|--------------|-------|")

        for r in aime_summary:
            status = "✅" if r.get("is_correct") else "❌"
            retry = []

            if r.get("note_was_retry"):
                retry.append("note")

            if r.get("answer_was_retry"):
                retry.append("answer")

            retry_text = ", ".join(retry) if retry else ""

            lines.append(
                f"| {r.get('id')} | {status} | {r.get('got')} | {r.get('expected')} | "
                f"{r.get('note_truncated')} | {r.get('note_soft_capitulation')} | "
                f"{r.get('answer_truncated')} | {retry_text} |"
            )
    else:
        lines.append("Brak wyników AIME.")

    lines.append("")
    write_text(report_path, "\n".join(lines).strip() + "\n")


# ============================================================
# CLI / MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="AIME 2026 reference eval — Ania 11B Research Preview")

    parser.add_argument("--model", default=DEFAULT_CONFIG["MODEL_PATH"])
    parser.add_argument("--aime", default=DEFAULT_CONFIG["AIME_FILE"])
    parser.add_argument("--output-root", default=DEFAULT_CONFIG["OUTPUT_ROOT"])


    parser.add_argument("--limit-aime", type=int, default=0,
                        help="Weź tylko pierwsze N zadań / take only the first N tasks")
    parser.add_argument("--skip-aime", type=int, default=0,
                        help="Pomiń pierwsze N zadań (druga partia) / skip first N tasks (second batch)")

    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_CONFIG["MAX_SEQ_LENGTH"])

    parser.add_argument("--gpus", default=None,
                        help="Indeksy kart GPU, np. '0' albo '1,3'. Bez flagi: wszystkie widoczne.")
    parser.add_argument("--no-tf32", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    config = dict(DEFAULT_CONFIG)
    config["MODEL_PATH"] = args.model
    config["AIME_FILE"] = args.aime
    config["OUTPUT_ROOT"] = args.output_root
    config["MAX_SEQ_LENGTH"] = args.max_seq_length

    if args.no_resume:
        config["RESUME"] = False

    if args.no_tf32:
        config["ALLOW_TF32"] = False

    output_dir = get_output_dir_name(config["MODEL_PATH"], config["OUTPUT_ROOT"])

    print("=" * 80)
    print("AIME 2026 — REFERENCYJNA EWALUACJA / REFERENCE EVALUATION")
    print("=" * 80)
    print(f"Model: {config['MODEL_PATH']}")
    print(f"Output: {output_dir}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"MAX_SEQ_LENGTH: {config['MAX_SEQ_LENGTH']}")
    print(f"AIME note budget: {config['AIME_NOTE_MAX_TOKENS']} (retry {config['AIME_NOTE_RETRY_MAX_TOKENS']})")
    print(f"ALLOW_TF32: {config['ALLOW_TF32']}")
    print("=" * 80)

    torch.backends.cuda.matmul.allow_tf32 = bool(config.get("ALLOW_TF32", True))
    torch.backends.cudnn.allow_tf32 = bool(config.get("ALLOW_TF32", True))

    set_seed(config["SEED"])

    write_json(output_dir / "config_snapshot.json", config)

    print("Ładowanie modelu...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["MODEL_PATH"],
        dtype=config["DTYPE"],
        load_in_4bit=False,
        max_seq_length=config["MAX_SEQ_LENGTH"],
        device_map="auto",
    )

    ensure_tokenizer(tokenizer)
    FastLanguageModel.for_inference(model)

    print("Model gotowy.")
    print("=" * 80)

    aime_dataset = load_aime_dataset(config["AIME_FILE"])

    if not aime_dataset:
        print("Brak zadań AIME — sprawdź ścieżkę --aime (patrz docstring: format JSONL).")
        return

    # --skip-aime N: pomiń pierwsze N zadań (druga partia z jednego pliku).
    # --skip-aime N: skip the first N tasks (second batch from one file).
    if args.skip_aime and args.skip_aime > 0:
        aime_dataset = aime_dataset[args.skip_aime :]

    if args.limit_aime and args.limit_aime > 0:
        aime_dataset = aime_dataset[: args.limit_aime]

    print(f"AIME: {len(aime_dataset)} zadań")
    aime_summary = run_aime(model, tokenizer, aime_dataset, output_dir, config)

    build_final_report(
        output_dir=output_dir,
        config=config,
        aime_summary=aime_summary,
    )

    deep_cleanup()

    print("=" * 80)
    print("KONIEC TESTU")
    print(f"Wyniki zapisane w: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
