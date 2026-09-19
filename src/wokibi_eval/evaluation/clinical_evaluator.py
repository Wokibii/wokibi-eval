"""Semantic clinical evaluator for original vs processed text pairs.

Independent of any LLM/model used to produce the processed text.
Three pillars: (1) spaCy + heuristic NER with semantic matching,
(2) LLM-as-a-Judge clinical rubric, (3) BioBERTpt global similarity.

Requires:
  - python -m spacy download pt_core_news_lg
  - sentence-transformers pulls pucpr/biobertpt-all on first load
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from wokibi_ai.prompt_manager import PromptManager
from wokibi_ai.llm import LLM
from wokibi_eval.evaluation.prompt_paths import PROMPTS_DIR as _PROMPTS_DIR

logger = logging.getLogger(__name__)

_SPACY_MODEL = "pt_core_news_lg"
_EMBEDDING_MODEL = "pucpr/biobertpt-all"
_JUDGE_PROMPT_VERSION = "2026-08-30"
_DEFAULT_NER_THRESHOLD = 0.75

_NUMERIC_RE = re.compile(r"\d+(?:[.,]\d+)?")
_DOSE_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|g|ml|ui\.?|meq|mmol|%|cpr|amp|gts?)\b",
    re.IGNORECASE,
)
_ROUTE_NEAR_DOSE_RE = re.compile(
    r"(?:\b(?:VO|EV|SC|IM|SL|VR)\b\s*)?"
    r"\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|g|ml|ui\.?|meq|mmol|%|cpr|amp|gts?)"
    r"(?:\s*\b(?:VO|EV|SC|IM|SL|VR)\b)?",
    re.IGNORECASE,
)

_CLINICAL_ACRONYMS = frozenset({
    "has", "dm", "irc", "drc", "itu", "pa", "fc", "fr", "sato2", "spo2",
    "po", "bav", "iam", "avc", "dpoc", "icc", "tep", "tvp", "ira", "dra",
    "hpb", "ca", "ca.", "hiv", "hbv", "hcv", "dm2", "dm1", "hasc", "eva",
    "glasgow", "apache", "sofa", "news", "mews", "pcr", "vsh", "inr", "ttpa",
    "hba1c", "tsh", "tgo", "tgp", "ureia", "cr", "tfge", "kdigo", "aki",
    "ckd", "hd", "dp", "avf", "cvc", "sng", "sne", "vm", "vni", "o2",
    "vo", "ev", "sc", "im", "sl", "vr",
})

_ACRONYM_TOKEN_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9.]{1,7})\b")

_COMPRESSION_FALLBACK: dict[str, Any] = {
    "original_char_len": 0,
    "processed_char_len": 0,
    "compression_rate": 0.0,
    "original_word_len": 0,
    "processed_word_len": 0,
    "word_compression_rate": 0.0,
    "is_empty_processed": False,
    "is_identical": False,
}

_NUMERIC_FALLBACK: dict[str, Any] = {
    "numeric_preservation_rate": 0.0,
}

_NER_FALLBACK: dict[str, Any] = {
    "ner_entities_original_count": 0,
    "ner_entities_processed_count": 0,
    "ner_recall": 0.0,
    "ner_precision": 0.0,
    "ner_f1_score": 0.0,
    "ner_similarity_threshold": _DEFAULT_NER_THRESHOLD,
}

_BIOBERT_FALLBACK: dict[str, Any] = {
    "biobert_similarity": 0.0,
}

_JUDGE_FALLBACK: dict[str, Any] = {
    "score_faithfulness": 0.0,
    "score_completeness": 0.0,
    "score_readability": 0.0,
}


def _safe_compression_rate(original_len: int, processed_len: int) -> float:
    if original_len <= 0:
        return 0.0
    return round(1.0 - (processed_len / original_len), 4)


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _clamp_score(value: Any, lo: int = 1, hi: int = 5) -> float:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0.0
    return float(max(lo, min(hi, score)))


class ClinicalEvaluator:
    """Evaluate a processed text against its original counterpart.

    Models are loaded once at init for efficient batch evaluation.
    """

    def __init__(
        self,
        spacy_model: str = _SPACY_MODEL,
        embedding_model: str = _EMBEDDING_MODEL,
        judge_llm: LLM | None = None,
        enable_judge: bool = True,
        ner_similarity_threshold: float = _DEFAULT_NER_THRESHOLD,
    ) -> None:
        self.nlp = spacy.load(spacy_model)
        self.embedder = SentenceTransformer(embedding_model)
        self.judge_llm = judge_llm
        self.enable_judge = enable_judge
        self.ner_similarity_threshold = ner_similarity_threshold
        self._judge_prompt_template: str | None = None

    def evaluate(self, original_text: str, processed_text: str) -> dict[str, Any]:
        """Return a flat metrics dict suitable for ``pd.DataFrame([row])``."""
        original = original_text if isinstance(original_text, str) else ""
        processed = processed_text if isinstance(processed_text, str) else ""

        row: dict[str, Any] = {}
        row.update(self._run_pipeline(
            "_compute_compression",
            self._compute_compression,
            original,
            processed,
            _COMPRESSION_FALLBACK,
        ))
        row.update(self._run_pipeline(
            "_compute_numeric_preservation",
            self._compute_numeric_preservation,
            original,
            processed,
            _NUMERIC_FALLBACK,
        ))
        row.update(self._run_pipeline(
            "_evaluate_ner_semantic",
            self._evaluate_ner_semantic,
            original,
            processed,
            {**_NER_FALLBACK, "ner_similarity_threshold": self.ner_similarity_threshold},
        ))
        row.update(self._run_pipeline(
            "_compute_biobert_similarity",
            self._compute_biobert_similarity,
            original,
            processed,
            _BIOBERT_FALLBACK,
        ))
        row.update(self._run_pipeline(
            "_evaluate_llm_judge",
            self._evaluate_llm_judge,
            original,
            processed,
            _JUDGE_FALLBACK,
        ))
        return row

    def _run_pipeline(
        self,
        name: str,
        fn: Callable[[str, str], dict[str, Any]],
        original: str,
        processed: str,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return fn(original, processed)
        except Exception:
            logger.exception("ClinicalEvaluator pipeline %s failed", name)
            return dict(fallback)

    def _compute_compression(
        self, original: str, processed: str
    ) -> dict[str, Any]:
        original_char_len = len(original)
        processed_char_len = len(processed)
        original_words = _tokenize(original)
        processed_words = _tokenize(processed)

        return {
            "original_char_len": original_char_len,
            "processed_char_len": processed_char_len,
            "compression_rate": _safe_compression_rate(
                original_char_len, processed_char_len
            ),
            "original_word_len": len(original_words),
            "processed_word_len": len(processed_words),
            "word_compression_rate": _safe_compression_rate(
                len(original_words), len(processed_words)
            ),
            "is_empty_processed": not processed.strip(),
            "is_identical": original.strip() == processed.strip(),
        }

    def _compute_numeric_preservation(
        self, original: str, processed: str
    ) -> dict[str, Any]:
        original_nums = _NUMERIC_RE.findall(original)
        if not original_nums:
            return {"numeric_preservation_rate": 1.0}

        processed_nums = set(_NUMERIC_RE.findall(processed))
        preserved = sum(1 for n in original_nums if n in processed_nums)
        return {
            "numeric_preservation_rate": round(preserved / len(original_nums), 4)
        }

    def _extract_entities(self, text: str) -> list[str]:
        if not text.strip():
            return []

        entities: list[str] = []

        doc = self.nlp(text)
        for ent in doc.ents:
            normalized = ent.text.strip().lower()
            if not normalized:
                continue
            if len(normalized) < 2 and normalized not in _CLINICAL_ACRONYMS:
                continue
            entities.append(normalized)

        for match in _ROUTE_NEAR_DOSE_RE.finditer(text):
            entities.append(match.group(0).strip().lower())
        for match in _DOSE_RE.finditer(text):
            entities.append(match.group(0).strip().lower())

        for match in _ACRONYM_TOKEN_RE.finditer(text):
            token = match.group(1).strip().lower().rstrip(".")
            if token in _CLINICAL_ACRONYMS:
                entities.append(token)

        return _dedupe_preserve_order(entities)

    def _semantic_entity_match(
        self,
        orig_ents: list[str],
        proc_ents: list[str],
    ) -> tuple[float, float, float]:
        if not orig_ents and not proc_ents:
            return 0.0, 0.0, 0.0
        if not orig_ents or not proc_ents:
            return 0.0, 0.0, 0.0

        o_emb = self.embedder.encode(
            orig_ents, convert_to_numpy=True, normalize_embeddings=True
        )
        p_emb = self.embedder.encode(
            proc_ents, convert_to_numpy=True, normalize_embeddings=True
        )
        sim = cosine_similarity(o_emb, p_emb)

        threshold = self.ner_similarity_threshold
        pairs = [
            (i, j, float(sim[i, j]))
            for i in range(len(orig_ents))
            for j in range(len(proc_ents))
            if float(sim[i, j]) >= threshold
        ]
        pairs.sort(key=lambda x: -x[2])

        matched_o: set[int] = set()
        matched_p: set[int] = set()
        for i, j, _ in pairs:
            if i in matched_o or j in matched_p:
                continue
            matched_o.add(i)
            matched_p.add(j)

        recall = len(matched_o) / len(orig_ents)
        precision = len(matched_p) / len(proc_ents)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return round(precision, 4), round(recall, 4), round(f1, 4)

    def _evaluate_ner_semantic(
        self, original: str, processed: str
    ) -> dict[str, Any]:
        orig_ents = self._extract_entities(original)
        proc_ents = self._extract_entities(processed)
        precision, recall, f1 = self._semantic_entity_match(orig_ents, proc_ents)

        return {
            "ner_entities_original_count": len(orig_ents),
            "ner_entities_processed_count": len(proc_ents),
            "ner_recall": recall,
            "ner_precision": precision,
            "ner_f1_score": f1,
            "ner_similarity_threshold": self.ner_similarity_threshold,
        }

    def _compute_biobert_similarity(
        self, original: str, processed: str
    ) -> dict[str, Any]:
        if not original.strip() or not processed.strip():
            return {"biobert_similarity": 0.0}

        embeddings = self.embedder.encode(
            [original, processed],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        score = float(cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1),
        )[0, 0])
        score = float(np.clip(score, -1.0, 1.0))
        return {"biobert_similarity": round(score, 4)}

    def _load_judge_prompt_template(self) -> str:
        if self._judge_prompt_template is None:
            pm = PromptManager(prompts_dir=_PROMPTS_DIR)
            self._judge_prompt_template = pm.load_prompt(
                "clinical_data",
                "clinical_judge",
                version=_JUDGE_PROMPT_VERSION,
                convert_to_template=False,
            )["template"]
        return self._judge_prompt_template

    def _parse_judge_json(self, raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Judge response has no JSON object")

        data = json.loads(text[start:end + 1])
        if not isinstance(data, dict):
            raise ValueError("Judge JSON root is not an object")

        return {
            "score_faithfulness": _clamp_score(data.get("score_faithfulness")),
            "score_completeness": _clamp_score(data.get("score_completeness")),
            "score_readability": _clamp_score(data.get("score_readability")),
        }

    def _evaluate_llm_judge(
        self, original: str, processed: str
    ) -> dict[str, Any]:
        if not self.enable_judge or self.judge_llm is None:
            return dict(_JUDGE_FALLBACK)

        template = self._load_judge_prompt_template()
        prompt = template.format(
            original_text=original,
            processed_text=processed,
        )
        self.judge_llm.start_chat([])
        raw = self.judge_llm.send_message(prompt)
        try:
            return self._parse_judge_json(raw if isinstance(raw, str) else "")
        except Exception:
            logger.debug(
                "Failed to parse judge JSON; raw=%r",
                (raw[:500] if isinstance(raw, str) else raw),
                exc_info=True,
            )
            return dict(_JUDGE_FALLBACK)
