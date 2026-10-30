"""Thin wrapper over free LLM APIs with DryRun support for SpendWise AI financial tools."""
import json
import os
import time

import config


class LLM:
    def __init__(self, provider=None):
        self.provider = provider or os.environ.get("AGENTX_PROVIDER") or (
            "gemini" if os.environ.get("GEMINI_API_KEY") else "groq" if os.environ.get("GROQ_API_KEY") else None)
        if self.provider is None:
            raise RuntimeError("No API key found. Set GEMINI_API_KEY or GROQ_API_KEY in .env (both are free).")
        if self.provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        else:
            from groq import Groq
            self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def generate(self, system, messages):
        """messages: list of {"role": "user"|"assistant", "content": str}. Returns text. Retries on 429."""
        for attempt in range(4):
            try:
                return self._call(system, messages)
            except Exception as e:
                converted = self._native_tool_call_as_json(e)
                if converted:
                    return converted
                # rate limit: back off and retry, otherwise surface the error
                if attempt < 3 and ("429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower()):
                    time.sleep(2 * 2 ** attempt)
                    continue
                raise

    @staticmethod
    def _native_tool_call_as_json(e):
        """Groq's gpt-oss sometimes emits a native tool call, which the API rejects (400 tool_use_failed).
        Turn that into our JSON action so the agent loop keeps going (unknown tool names are reported back)."""
        try:
            body = e.response.json()["error"]
            if body.get("code") != "tool_use_failed":
                return None
            call = json.loads(body["failed_generation"])
            return json.dumps({"thought": "(native tool call)", "action": str(call["name"]).split(".")[-1],
                               "args": call.get("arguments") or {}})
        except Exception:
            return None

    def _call(self, system, messages):
        if self.provider == "gemini":
            from google.genai import types
            text = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            r = self.client.models.generate_content(
                model=config.GEMINI_MODEL, contents=text,
                config=types.GenerateContentConfig(system_instruction=system, temperature=config.TEMPERATURE))
            return r.text or ""
        r = self.client.chat.completions.create(
            model=config.GROQ_MODEL, temperature=config.TEMPERATURE,
            messages=[{"role": "system", "content": system}] + messages)
        msg = r.choices[0].message
        # some Groq reasoning models (gpt-oss) leave `content` empty and put the reply in `reasoning`
        return msg.content or getattr(msg, "reasoning", None) or ""


class DryRunLLM:
    """Scripted stand-in so you can test your setup with NO API key: python main.py --dry-run "..."

    Simulates realistic ReAct reasoning, dynamic financial tool selection, multi-tool audit
    sequences, and error recovery without external network calls.
    """

    def __init__(self, replies=None):
        self.step = 0
        self.replies = list(replies) if replies is not None else None

    def generate(self, system, messages):
        self.step += 1
        if self.replies is not None:
            return self.replies.pop(0) if self.replies else '{"thought": "Sufficient data gathered.", "final_answer": "Dry run execution complete."}'

        # Extract user query
        query = ""
        for m in messages:
            if m.get("role") == "user":
                query = m.get("content", "").lower()
                break

        # Check for error recovery simulation
        if "error" in query:
            if self.step == 1:
                return '{"thought": "Attempting an unverified tool name.", "action": "non_existent_tool", "args": {}}'
            if self.step == 2:
                return '{"thought": "Tool failed with error. Recovering by calling categorize_transactions.", "action": "categorize_transactions", "args": {"filename": "transactions.csv", "limit": 3}}'
            return '{"thought": "Synthesizing after recovery.", "final_answer": "Recovered from initial tool failure and successfully categorized expenses."}'

        # Check for repeat tool call simulation
        if "repeat" in query:
            if self.step <= 3:
                return '{"thought": "Repeatedly inspecting files.", "action": "list_files", "args": {}}'
            return '{"thought": "Repeat limit hit; finalizing answer.", "final_answer": "Repeated tool test completed successfully."}'

        # Subscription analysis scenario
        if "sub" in query or "recur" in query:
            if self.step == 1:
                return '{"thought": "Analyze recurring payments and cadences.", "action": "detect_subscriptions", "args": {"filename": "transactions.csv", "min_cycles": 3}}'
            return '{"thought": "Synthesize subscription findings.", "final_answer": "Subscription analysis complete: Verified interval cadences, tracked price creep, and reported missing merchant fields."}'

        # Anomaly detection scenario
        if "anomal" in query or "surge" in query or "spike" in query:
            if self.step == 1:
                return '{"thought": "Detect statistical spending anomalies and surges.", "action": "detect_anomalies", "args": {"filename": "transactions.csv", "sensitivity": 2.5}}'
            return '{"thought": "Synthesize anomaly report.", "final_answer": "Spending anomaly hunt complete: Identified statistical outliers using category IQR fences and Z-scores without relying on fraud labels."}'

        # Duplicate detection scenario
        if "dup" in query or "double" in query:
            if self.step == 1:
                return '{"thought": "Detect duplicate payments in transaction log.", "action": "detect_duplicates", "args": {"filename": "transactions.csv", "window_hours": 24}}'
            return '{"thought": "Summarize duplicate payment audit.", "final_answer": "Duplicate payment analysis complete: Verified transaction IDs for ledger replays and checked time-windowed similar debits."}'

        # Budget planning scenario
        if "budget" in query or "plan" in query:
            if self.step == 1:
                return '{"thought": "Generate 50/30/20 budget allocations with user income.", "action": "plan_budget", "args": {"filename": "transactions.csv", "monthly_income": 60000.0, "fixed_commitments": 10000.0}}'
            return '{"thought": "Deliver personalized budget recommendations.", "final_answer": "Personalized 50/30/20 budget recommendation generated based on actual spending and user-provided income."}'

        # Categorization scenario
        if "categor" in query:
            if self.step == 1:
                return '{"thought": "Inspect transaction category breakdown.", "action": "categorize_transactions", "args": {"filename": "transactions.csv", "limit": 5}}'
            return '{"thought": "Synthesize categorization findings.", "final_answer": "Categorization complete: Analyzed transaction breakdown across primary spending categories based on verified transaction metadata."}'

        # Full multi-tool audit scenario
        if "audit" in query:
            if self.step == 1:
                return '{"thought": "Start financial audit by categorizing spending.", "action": "categorize_transactions", "args": {"filename": "transactions.csv", "limit": 5}}'
            if self.step == 2:
                return '{"thought": "Check for duplicate transactions or double debits.", "action": "detect_duplicates", "args": {"filename": "transactions.csv", "window_hours": 24}}'
            if self.step == 3:
                return '{"thought": "Check for spending surges and statistical outliers.", "action": "detect_anomalies", "args": {"filename": "transactions.csv", "sensitivity": 2.5}}'
            return '{"thought": "Synthesizing complete financial audit report.", "final_answer": "Comprehensive Financial Audit Complete: Evaluated transaction categories, confirmed duplicate safety, and flagged statistical spending outliers using registered tools."}'

        # Default starter-compatible scenario (list_files -> calculator -> final)
        if self.step == 1:
            return '{"thought": "See which data files exist.", "action": "list_files", "args": {}}'
        if self.step == 2:
            return '{"thought": "Check a calculation.", "action": "calculator", "args": {"expression": "2 + 2 * 10"}}'
        return '{"thought": "I have enough.", "final_answer": "Dry run OK: tools executed (list_files, calculator)."}'
