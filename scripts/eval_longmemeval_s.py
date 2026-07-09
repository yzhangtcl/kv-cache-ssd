#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _json_load(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"expected a list in {path}")
    return data


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        print(json.dumps(entry, ensure_ascii=False), file=f)


def _load_jsonl_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        qid = entry.get("question_id")
        if qid is not None:
            out[str(qid)] = entry
    return out


def _limit_per_type(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    counts: dict[str, int] = {}
    selected = []
    for row in rows:
        key = str(row.get("question_type", "unknown"))
        count = counts.get(key, 0)
        if count >= limit:
            continue
        selected.append(row)
        counts[key] = count + 1
    return selected


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:160] or "sample"


def _normalize_for_guardrail(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\b(user|assistant|system)\b", " ", text)
    text = re.sub(r"[^a-z0-9.%$:/-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _number_tokens(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z0-9])[$]?\d+(?:[.,]\d+)?%?(?![A-Za-z0-9])", text)


def _word_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", _normalize_for_guardrail(text))
        if token not in {"a", "an", "the"}
    ]


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    if not needle:
        return False
    cursor = 0
    for token in haystack:
        if token == needle[cursor]:
            cursor += 1
            if cursor == len(needle):
                return True
    return False


def _is_allowed_missing_suffix(missing: list[str]) -> bool:
    if not missing:
        return True
    suffix = " ".join(missing)
    if suffix in {"each way", "one way", "round trip"}:
        return True
    return missing[0] in {
        "in",
        "at",
        "from",
        "to",
        "for",
        "on",
        "downtown",
        "nearby",
        "local",
    }


def apply_judge_guardrails(
    *,
    question: str,
    answer: str,
    hypothesis: str,
    label: bool,
) -> tuple[bool, list[str]]:
    reasons = []
    answer_norm = _normalize_for_guardrail(answer)
    hypothesis_norm = _normalize_for_guardrail(hypothesis)
    if not answer_norm or not hypothesis_norm:
        return label, reasons

    answer_numbers = _number_tokens(answer_norm)
    hypothesis_numbers = _number_tokens(hypothesis_norm)
    if answer_numbers:
        if answer_numbers != hypothesis_numbers:
            return False, [f"number_mismatch:{answer_numbers}!={hypothesis_numbers}"]

    answer_words = _word_tokens(answer)
    hypothesis_words = _word_tokens(hypothesis)
    if len(answer_words) >= 2 and _is_subsequence(hypothesis_words, answer_words) and hypothesis_words != answer_words:
        coverage = len(hypothesis_words) / max(1, len(answer_words))
        missing_suffix = answer_words[len(hypothesis_words) :] if answer_words[: len(hypothesis_words)] == hypothesis_words else []
        proper_phrase = any(ch.isupper() for ch in answer) or any(
            keyword in question.lower()
            for keyword in (
                "name",
                "playlist",
                "play",
                "book",
                "movie",
                "song",
                "degree",
                "certification",
                "event",
                "brand",
                "breed",
            )
        )
        if _is_allowed_missing_suffix(missing_suffix):
            return label, reasons
        if proper_phrase or coverage < 0.75:
            return False, [f"incomplete_answer_span:{' '.join(hypothesis_words)}<{' '.join(answer_words)}"]

    if answer_norm not in hypothesis_norm and hypothesis_norm not in answer_norm:
        return label, reasons
    return label, reasons


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _make_progress_logger(path: Path | None):
    def log(event: str, payload: dict[str, Any]) -> None:
        record = {"time": _now(), "event": event, **payload}
        text = json.dumps(record, ensure_ascii=False)
        print(text, flush=True)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                print(text, file=f)

    return log


def _format_session(session: list[dict[str, Any]], *, user_only: bool) -> str:
    turns = []
    for turn in session:
        role = str(turn.get("role", "unknown"))
        if user_only and role != "user":
            continue
        content = str(turn.get("content", "")).strip()
        if content:
            turns.append({"role": role, "content": content})
    return json.dumps(turns, ensure_ascii=False)


def build_prompt(entry: dict[str, Any], *, history_format: str, user_only: bool) -> str:
    dates = entry.get("haystack_dates") or []
    sessions = entry.get("haystack_sessions") or []
    session_ids = entry.get("haystack_session_ids") or [None] * len(sessions)

    chunks = []
    for idx, session in enumerate(sessions):
        date = dates[idx] if idx < len(dates) else ""
        session_id = session_ids[idx] if idx < len(session_ids) else idx
        if history_format == "json":
            payload = {
                "session_id": session_id,
                "date": date,
                "turns": [
                    {
                        "role": turn.get("role", "unknown"),
                        "content": str(turn.get("content", "")).strip(),
                    }
                    for turn in session
                    if not user_only or turn.get("role") == "user"
                ],
            }
            chunks.append(json.dumps(payload, ensure_ascii=False))
        else:
            lines = [f"[Session {session_id} | {date}]"]
            lines.append(_format_session(session, user_only=user_only))
            chunks.append("\n".join(lines))

    history = "\n".join(chunks)
    question_date = entry.get("question_date", "")
    question = entry.get("question", "")
    return (
        "You are given timestamped chat history between a user and an assistant. "
        "Answer the final question using only the information in the history. "
        "This is a short extraction task. Return only the minimal complete answer span, not a sentence, unless "
        "a sentence is the only natural answer. Do not explain your reasoning. Do not include role names, JSON, "
        "XML tags, thinking text, or repeated answers. Do not output a partial phrase if a fuller name, title, "
        "duration, date, quantity, color description, location, organization, or event name is available. "
        "If the user asks where, what place, which store, which class, which shade, or which event, answer only "
        "that exact requested item. If the history contains a relevant fact, answer it directly and do not hedge. "
        "If the history truly does not contain the requested information, say only: Not available.\n\n"
        f"Question date: {question_date}\n\n"
        "Chat history:\n"
        f"{history}\n\n"
        f"Question: {question}\n\n"
        "Complete answer only:"
    )


def get_anscheck_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    *,
    abstention: bool,
    lenient: bool,
) -> str:
    verdict_line = "Reply with exactly one word: yes or no."
    lenient_rule = (
        "Use a lenient short-answer criterion: answer yes if the response captures the main requested "
        "entity, value, date, duration, quantity, place, color, person, organization, or event from the "
        "reference answer, even if it omits minor qualifiers or is embedded in a short sentence. "
        "Answer no if it names a different entity/value, is too vague to identify the reference answer, "
        "or adds extra conflicting answers. For example, '45 minutes' matches '45 minutes each way', "
        "but 'Business' does not match 'Business Administration' because it is too generic.\n\n"
    )
    if abstention:
        return (
            "Judge whether the model correctly says the question cannot be answered from the provided memory.\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\n"
            f"{verdict_line}"
        )
    if task in {"single-session-user", "single-session-assistant", "multi-session"}:
        return (
            (
                lenient_rule
                if lenient
                else "Judge whether the model response fully answers the question with the same meaning as the reference answer.\n\n"
            )
            + f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            f"{verdict_line}"
        )
    if task == "temporal-reasoning":
        return (
            (
                lenient_rule + "For day-count answers, also allow an off-by-one error.\n\n"
                if lenient
                else "Judge whether the response matches the reference answer. For day-count answers, allow an off-by-one error.\n\n"
            )
            + f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            f"{verdict_line}"
        )
    if task == "knowledge-update":
        return (
            (
                lenient_rule + "If older information is also mentioned, answer yes only when the latest correct answer is still clear.\n\n"
                if lenient
                else "Judge whether the response contains the latest correct answer, even if it also mentions older information.\n\n"
            )
            + f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            f"{verdict_line}"
        )
    if task == "single-session-preference":
        return (
            (
                "Use a lenient personalized-answer criterion: answer yes if the response recalls and uses the user's "
                "main relevant preference correctly, even if it omits minor rubric details. Answer no if it uses a "
                "different or conflicting preference.\n\n"
                if lenient
                else "Judge whether the response uses the user's personal preference correctly according to the rubric.\n\n"
            )
            + f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
            f"{verdict_line}"
        )
    raise NotImplementedError(f"unsupported question_type: {task}")


def deepseek_chat(
    prompt: str,
    *,
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
    max_retries: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 10,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return str(data["choices"][0]["message"]["content"]).strip()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"DeepSeek API request failed: {last_error}")


def judge_with_deepseek(
    entry: dict[str, Any],
    hypothesis: str,
    *,
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
    max_retries: int,
    lenient: bool,
    guardrails: bool,
) -> dict[str, Any]:
    qid = str(entry["question_id"])
    prompt = get_anscheck_prompt(
        str(entry["question_type"]),
        str(entry["question"]),
        str(entry["answer"]),
        hypothesis,
        abstention="_abs" in qid,
        lenient=lenient,
    )
    response = deepseek_chat(
        prompt,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    raw_label = "yes" in response.lower()
    label = raw_label
    guardrail_reasons: list[str] = []
    if guardrails:
        label, guardrail_reasons = apply_judge_guardrails(
            question=str(entry["question"]),
            answer=str(entry["answer"]),
            hypothesis=hypothesis,
            label=raw_label,
        )
    return {
        "model": model,
        "lenient": lenient,
        "guardrails": guardrails,
        "raw_response": response,
        "raw_label": raw_label,
        "guardrail_reasons": guardrail_reasons,
        "label": label,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SSD KV-cache generation on LongMemEval-S.")
    parser.add_argument("--data-file", default="data/longmemeval_s_cleaned.json")
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "/root/blockdata/data/models/qwen3.5-9b"))
    parser.add_argument("--ssd-dir", default=os.environ.get("SSD_DIR", "/root/blockdata/data/kvssd-longmemeval"))
    parser.add_argument("--cleanup-ssd-cache", action="store_true", help="Delete each question's SSD KV cache after it finishes.")
    parser.add_argument("--out-file", default="outputs/longmemeval_s_ssd_predictions.jsonl")
    parser.add_argument("--eval-log", default=None)
    parser.add_argument("--progress-log", default=None, help="Optional jsonl file for per-question progress events.")
    parser.add_argument("--progress-every", type=int, default=10, help="Log decode progress every N tokens.")
    parser.add_argument("--limit", type=int, default=0, help="0 means all examples.")
    parser.add_argument("--limit-per-type", type=int, default=0, help="0 means no per-question_type limit.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--top-k-blocks", type=int, default=1024)
    parser.add_argument("--summary-centroids-per-block", type=int, default=4)
    parser.add_argument("--history-format", choices=("json", "nl"), default="json")
    parser.add_argument("--user-only", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--enable-thinking", choices=("auto", "0", "1"), default="auto")
    parser.add_argument("--judge", action="store_true", help="Call DeepSeek API and write eval labels.")
    parser.add_argument("--judge-lenient", action="store_true", help="Use a looser short-answer equivalence prompt for DeepSeek judging.")
    parser.add_argument("--no-judge-guardrails", action="store_true", help="Disable deterministic guardrails for obvious judge mistakes.")
    parser.add_argument("--judge-existing", action="store_true", help="Only judge existing predictions.")
    parser.add_argument("--deepseek-api-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    parser.add_argument("--deepseek-model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--deepseek-base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--deepseek-timeout", type=int, default=60)
    parser.add_argument("--deepseek-max-retries", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_file = Path(args.data_file)
    out_file = Path(args.out_file)
    eval_log = Path(args.eval_log) if args.eval_log else Path(str(out_file) + ".deepseek.log")
    progress_log = Path(args.progress_log) if args.progress_log else None
    log_progress = _make_progress_logger(progress_log)
    data = _json_load(data_file)
    subset = data[args.start :]
    subset = _limit_per_type(subset, args.limit_per_type)
    if args.limit > 0:
        subset = subset[: args.limit]
    type_counts: dict[str, int] = {}
    for entry in subset:
        key = str(entry.get("question_type", "unknown"))
        type_counts[key] = type_counts.get(key, 0) + 1
    log_progress(
        "dataset_selected",
        {
            "total": len(subset),
            "start": args.start,
            "limit": args.limit,
            "limit_per_type": args.limit_per_type,
            "type_counts": type_counts,
        },
    )

    predictions = _load_jsonl_by_qid(out_file)
    evals = _load_jsonl_by_qid(eval_log)
    if args.judge_existing and not predictions:
        log_progress(
            "warning_no_existing_predictions",
            {
                "out_file": str(out_file),
                "message": "--judge-existing was set, but no predictions were loaded from --out-file.",
            },
        )

    if args.judge_existing:
        model = None
        tokenizer = None
        SSDBlockKVConfig = None
        generate_with_ssd_block_kv = None
    else:
        from model_loader import load_tokenizer_and_model
        from ssd_block_kvcache import SSDBlockKVConfig, generate_with_ssd_block_kv

        tokenizer, model = load_tokenizer_and_model(args.model_dir)

    if args.judge and not args.deepseek_api_key:
        raise RuntimeError("set DEEPSEEK_API_KEY or pass --deepseek-api-key to use --judge")

    correct = 0
    judged = 0
    for idx, entry in enumerate(subset, start=args.start):
        qid = str(entry["question_id"])
        question_ssd_dir = Path(args.ssd_dir) / _safe_name(qid)
        hypothesis = None
        log_progress(
            "question_start",
            {
                "index": idx,
                "question_id": qid,
                "question_type": entry.get("question_type"),
                "question": entry.get("question"),
            },
        )
        if qid in predictions:
            hypothesis = str(predictions[qid].get("hypothesis", ""))
            print(f"[{idx}] skip generated {qid}")
        elif args.judge_existing:
            log_progress(
                "skip_missing_prediction",
                {
                    "index": idx,
                    "question_id": qid,
                    "out_file": str(out_file),
                },
            )
        elif not args.judge_existing:
            assert model is not None and tokenizer is not None
            prompt = build_prompt(entry, history_format=args.history_format, user_only=args.user_only)
            prompt_tokens = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            log_progress(
                "prompt_built",
                {
                    "index": idx,
                    "question_id": qid,
                    "prompt_chars": len(prompt),
                    "prompt_tokens_before_chat_template": prompt_tokens,
                    "sessions": len(entry.get("haystack_sessions") or []),
                },
            )
            thinking = None if args.enable_thinking == "auto" else args.enable_thinking == "1"

            def generation_progress(event: str, payload: dict[str, Any]) -> None:
                if event in {"decode_cache_built", "decode_forward_done"}:
                    step = int(payload.get("step", 0) or 0)
                    every = max(1, int(args.progress_every))
                    if step > 1 and step % every != 0:
                        return
                log_progress(
                    event,
                    {
                        "index": idx,
                        "question_id": qid,
                        **payload,
                    },
                )

            result = generate_with_ssd_block_kv(
                model,
                tokenizer,
                prompt,
                ssd_dir=question_ssd_dir,
                max_new_tokens=args.max_new_tokens,
                prefill_chunk_tokens=args.prefill_chunk_tokens,
                config=SSDBlockKVConfig(
                    block_size=args.block_size,
                    top_k_blocks=args.top_k_blocks,
                    summary_centroids_per_block=args.summary_centroids_per_block,
                    preserve_original_positions=True,
                ),
                use_chat_template=not args.no_chat_template,
                chat_template_enable_thinking=thinking,
                progress_callback=generation_progress,
            )
            hypothesis = result.text.strip()
            _append_jsonl(out_file, {"question_id": qid, "hypothesis": hypothesis})
            predictions[qid] = {"question_id": qid, "hypothesis": hypothesis}
            log_progress(
                "question_generated",
                {
                    "index": idx,
                    "question_id": qid,
                    "generated_tokens": result.generated_tokens,
                    "elapsed_sec": round(result.elapsed_sec, 3),
                    "peak_memory_gb": round(result.peak_memory_gb, 3),
                    "blocks_written": result.ssd_cache.blocks_written,
                    "tokens_loaded": result.ssd_cache.tokens_loaded,
                    "output_chars": len(hypothesis),
                },
            )
            print(
                f"[{idx}] generated {qid} tokens={result.generated_tokens} "
                f"sec={result.elapsed_sec:.2f} blocks={result.ssd_cache.blocks_written}"
            )

        if args.judge and qid not in evals and hypothesis is not None:
            log_progress("judge_start", {"index": idx, "question_id": qid})
            label = judge_with_deepseek(
                entry,
                hypothesis,
                api_key=args.deepseek_api_key,
                model=args.deepseek_model,
                base_url=args.deepseek_base_url,
                timeout=args.deepseek_timeout,
                max_retries=args.deepseek_max_retries,
                lenient=args.judge_lenient,
                guardrails=not args.no_judge_guardrails,
            )
            log_entry = {
                "question_id": qid,
                "hypothesis": hypothesis,
                "answer": entry.get("answer"),
                "question_type": entry.get("question_type"),
                "autoeval_label": label,
            }
            _append_jsonl(eval_log, log_entry)
            evals[qid] = log_entry
            log_progress(
                "judge_done",
                {
                    "index": idx,
                    "question_id": qid,
                    "label": label["label"],
                    "raw_response": label["raw_response"],
                },
            )
            print(f"[{idx}] judged {qid}: {label['label']} ({label['raw_response']})")

        if qid in evals:
            judged += 1
            if bool(evals[qid].get("autoeval_label", {}).get("label")):
                correct += 1
        log_progress(
            "question_done",
            {
                "index": idx,
                "question_id": qid,
                "judged": qid in evals,
                "running_correct": correct,
                "running_judged": judged,
            },
        )
        if args.cleanup_ssd_cache and question_ssd_dir.exists():
            shutil.rmtree(question_ssd_dir)
            log_progress(
                "ssd_cache_cleaned",
                {
                    "index": idx,
                    "question_id": qid,
                    "path": str(question_ssd_dir),
                },
            )

    print(f"predictions: {out_file}")
    if args.judge:
        acc = correct / judged if judged else 0.0
        print(f"deepseek_eval_log: {eval_log}")
        print(f"deepseek_accuracy: {acc:.4f} ({correct}/{judged})")


if __name__ == "__main__":
    main()
