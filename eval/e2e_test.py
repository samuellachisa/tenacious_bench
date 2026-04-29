"""
e2e_test.py — Full end-to-end integration test for Tenacious Agent.

Tests the complete pipeline in order:
  1. GET /health
  2. POST /prospect (DataFlow Technologies)
  3. POST /webhook/email/reply
  4. POST /webhook/sms
  5. POST /webhook/cal

Saves results to eval/e2e_test_results.json.
Prints final summary with HubSpot and Langfuse check reminders.

Usage:
    # Start the server first:
    uvicorn agent.main:app --reload --port 8000

    # Then run:
    python eval/e2e_test.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

BASE_URL = "https://tenacious-agent.onrender.com"
RESULTS_FILE = Path(__file__).parent / "e2e_test_results.json"

# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

SYNTHETIC_PROSPECT = {
    "company_name": "DataFlow Technologies",
    "contact_email": "iamsamuellachisa@gmail.com",
    "contact_first_name": "SAMUEL",
    "contact_last_name": "LACHISA",
    "phone_number": "+251923393204",
    "timezone": "America/Los_Angeles",
}

SYNTHETIC_REPLY = {
    "from_email": "iamsamuellachisa@gmail.com",
    "from_name": "ABEBE BIKILA",
    "company_name": "DataFlow Technologies",
    "message_preview": "Hi, thanks for reaching out. I'd be happy to learn more.",
    "phone_number": "+251923393204",
}

SYNTHETIC_SMS = {
    "from_number": "+251923393204",
    "to_number": "+251923393204",
    "text": "Yes, I'd like to schedule a call",
    "date": datetime.now(timezone.utc).isoformat(),
}

SYNTHETIC_CAL_BOOKING = {
    "triggerEvent": "BOOKING_CREATED",
    "payload": {
        "title": "Discovery Call — DataFlow Technologies",
        "startTime": "2026-04-29T10:00:00Z",
        "endTime": "2026-04-29T10:30:00Z",
        "attendees": [
            {
                "name": "CHALA BIKILA",
                "email": "iamsamuellachisa@gmail.com",
                "timeZone": "America/Los_Angeles",
            }
        ],
        "organizer": {
            "name": "Tenacious Consulting",
            "email": "outbound@tenacious.consulting",
        },
        "uid": "synthetic-booking-001",
        "status": "ACCEPTED",
    },
}


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

async def run_e2e_tests() -> dict[str, Any]:
    results: dict[str, Any] = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "steps": [],
        "passed": 0,
        "failed": 0,
        "total": 0,
    }

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:

        # ------------------------------------------------------------------
        # Step 1: GET /health
        # ------------------------------------------------------------------
        step = await _run_step(
            client=client,
            step_num=1,
            name="GET /health",
            method="GET",
            path="/health",
            expected_status=200,
            assertions=[
                lambda r: r.json().get("status") == "ok",
                lambda r: "outbound_enabled" in r.json(),
                lambda r: "timestamp" in r.json(),
            ],
            assertion_labels=[
                "status == ok",
                "outbound_enabled present",
                "timestamp present",
            ],
        )
        results["steps"].append(step)
        _tally(results, step)

        # ------------------------------------------------------------------
        # Step 2: POST /prospect — DataFlow Technologies
        # ------------------------------------------------------------------
        step = await _run_step(
            client=client,
            step_num=2,
            name="POST /prospect (DataFlow Technologies)",
            method="POST",
            path="/prospect",
            body=SYNTHETIC_PROSPECT,
            expected_status=200,
            assertions=[
                lambda r: r.json().get("status") == "pipeline_queued",
                lambda r: r.json().get("company_name") == "DataFlow Technologies",
                lambda r: "contact_email" in r.json(),
            ],
            assertion_labels=[
                "status == pipeline_queued",
                "company_name == DataFlow Technologies",
                "contact_email present",
            ],
        )
        results["steps"].append(step)
        _tally(results, step)

        print("  Waiting 5s for background pipeline tasks...")
        await asyncio.sleep(5)

        # ------------------------------------------------------------------
        # Step 3: POST /webhook/email/reply
        # ------------------------------------------------------------------
        step = await _run_step(
            client=client,
            step_num=3,
            name="POST /webhook/email/reply",
            method="POST",
            path="/webhook/email/reply",
            body=SYNTHETIC_REPLY,
            expected_status=200,
            assertions=[
                lambda r: r.json().get("received") is True,
                lambda r: r.json().get("from_email") == "iamsamuellachisa@gmail.com",
                lambda r: r.json().get("pipeline") == "queued",
            ],
            assertion_labels=[
                "received == True",
                "from_email correct",
                "pipeline == queued",
            ],
        )
        results["steps"].append(step)
        _tally(results, step)

        print("  Waiting 15s for reply pipeline tasks...")
        await asyncio.sleep(15)

        # ------------------------------------------------------------------
        # Step 4: POST /webhook/sms
        # ------------------------------------------------------------------
        step = await _run_step(
            client=client,
            step_num=4,
            name="POST /webhook/sms (warm lead scheduling)",
            method="POST",
            path="/webhook/sms",
            body=SYNTHETIC_SMS,
            expected_status=200,
            assertions=[
                lambda r: r.json().get("received") is True,
                lambda r: r.json().get("from_number") == "+251923393204",
                lambda r: r.json().get("action") == "scheduling_queued",
            ],
            assertion_labels=[
                "received == True",
                "from_number correct",
                "action == scheduling_queued",
            ],
        )
        results["steps"].append(step)
        _tally(results, step)

        # ------------------------------------------------------------------
        # Step 5: POST /webhook/cal — BOOKING_CREATED
        # ------------------------------------------------------------------
        step = await _run_step(
            client=client,
            step_num=5,
            name="POST /webhook/cal (BOOKING_CREATED)",
            method="POST",
            path="/webhook/cal",
            body=SYNTHETIC_CAL_BOOKING,
            expected_status=200,
            assertions=[
                lambda r: r.json().get("received") is True,
                lambda r: r.json().get("trigger_event") == "BOOKING_CREATED",
                lambda r: "booking_processing_queued" in r.json().get("action", ""),
            ],
            assertion_labels=[
                "received == True",
                "trigger_event == BOOKING_CREATED",
                "action == booking_processing_queued",
            ],
        )
        results["steps"].append(step)
        _tally(results, step)

    # Save results
    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _run_step(
    client: httpx.AsyncClient,
    step_num: int,
    name: str,
    method: str,
    path: str,
    expected_status: int,
    assertions: list,
    assertion_labels: list[str],
    body: dict | None = None,
) -> dict[str, Any]:
    print(f"\nStep {step_num}: {name}")
    print(f"  {method} {path}")

    step_result: dict[str, Any] = {
        "step": step_num,
        "name": name,
        "method": method,
        "path": path,
        "passed": False,
        "http_status": None,
        "response_body": None,
        "assertion_results": [],
        "error": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        if method == "GET":
            response = await client.get(path)
        elif method == "POST":
            response = await client.post(path, json=body)
        else:
            raise ValueError(f"Unsupported method: {method}")

        step_result["http_status"] = response.status_code
        try:
            step_result["response_body"] = response.json()
        except Exception:
            step_result["response_body"] = response.text

        print(f"  HTTP {response.status_code}")

        # Status assertion
        status_ok = response.status_code == expected_status
        if not status_ok:
            print(f"  FAIL: expected HTTP {expected_status}, got {response.status_code}")
            step_result["error"] = f"HTTP {response.status_code} != {expected_status}"
            return step_result

        # Custom assertions
        all_passed = True
        for fn, label in zip(assertions, assertion_labels):
            try:
                passed = fn(response)
            except Exception as exc:
                passed = False
                print(f"  FAIL assertion '{label}': {exc}")

            step_result["assertion_results"].append(
                {"label": label, "passed": passed}
            )
            status_icon = "PASS" if passed else "FAIL"
            print(f"  {status_icon}: {label}")
            if not passed:
                all_passed = False

        step_result["passed"] = all_passed

    except httpx.ConnectError:
        step_result["error"] = (
            f"Connection refused at {BASE_URL}. "
            "Is the server running? Run: uvicorn agent.main:app --reload --port 8000"
        )
        print(f"  ERROR: {step_result['error']}")

    except Exception as exc:
        step_result["error"] = str(exc)
        print(f"  ERROR: {exc}")

    return step_result


def _tally(results: dict, step: dict) -> None:
    results["total"] += 1
    if step.get("passed"):
        results["passed"] += 1
    else:
        results["failed"] += 1


def _print_summary(results: dict) -> None:
    passed = results["passed"]
    failed = results["failed"]
    total = results["total"]

    print(f"\n{'='*60}")
    print("E2E Test Summary")
    print(f"{'='*60}")
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")
    print(f"  Results saved to: {RESULTS_FILE}")
    print(f"\n{'='*60}")
    print("Post-run verification checklist:")
    print("  HubSpot: Check that jordan.lee@dataflow.tech contact was created")
    print("    with icp_segment, ai_maturity_score, and hs_lead_status fields.")
    print("  HubSpot: Verify a deal was created linked to the contact.")
    print("  Langfuse: Check traces for enrichment_pipeline_complete,")
    print("    qualifier_result, email_sink (or email_sent), and")
    print("    calcom_booking_created events.")
    print("  Cal.com: Verify booking attempt was logged (sandbox mode).")
    print(f"{'='*60}\n")

    if failed > 0:
        print(f"WARNING: {failed} step(s) failed. Check output above.")
        sys.exit(1)
    else:
        print("All steps passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_e2e_tests())
