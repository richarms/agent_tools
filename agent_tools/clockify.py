import datetime as dt
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


DEFAULT_BASE_URL = "https://api.clockify.me/api/v1"
DEFAULT_TIMEZONE = "UTC"


class ClockifyError(Exception):
    """Raised when the Clockify API returns an error response."""


def _resolve_timezone(name: Optional[str] = None) -> dt.tzinfo:
    tz_name = (name or os.getenv("CLOCKIFY_TIMEZONE") or DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    if ZoneInfo:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass
    return dt.timezone.utc


def _parse_datetime(value: str, tz: dt.tzinfo) -> dt.datetime:
    cleaned = value.replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tz)
    return parsed


def _isoformat(value: dt.datetime) -> str:
    return value.isoformat()


def _next_monday(today: Optional[dt.date] = None) -> dt.date:
    base = today or dt.date.today()
    offset = (7 - base.weekday()) % 7
    if offset == 0:
        offset = 7
    return base + dt.timedelta(days=offset)


def _extract_tag_ids(entry: Dict[str, Any]) -> List[str]:
    if "tagIds" in entry and isinstance(entry["tagIds"], list):
        return [str(tag_id) for tag_id in entry["tagIds"]]
    tags = entry.get("tags") or []
    tag_ids: List[str] = []
    for tag in tags:
        tag_id = tag.get("id")
        if tag_id:
            tag_ids.append(str(tag_id))
    return tag_ids


def _extract_interval(entry: Dict[str, Any], tz: dt.tzinfo) -> Optional[Tuple[dt.datetime, dt.datetime]]:
    interval = entry.get("timeInterval") or {}
    start_raw = interval.get("start")
    end_raw = interval.get("end")
    if not start_raw or not end_raw:
        return None
    try:
        start = _parse_datetime(start_raw, tz)
        end = _parse_datetime(end_raw, tz)
    except Exception:
        return None
    if end <= start:
        return None
    return start, end


@dataclass
class EntrySnapshot:
    entry: Dict[str, Any]
    start: dt.datetime
    end: dt.datetime

    @property
    def duration(self) -> dt.timedelta:
        return self.end - self.start


class ClockifyClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL):
        normalized = (api_key or "").strip()
        if not normalized:
            raise ClockifyError("CLOCKIFY_API_KEY is not set")
        self.api_key = normalized
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        headers = kwargs.pop("headers", {}) or {}
        headers["X-Api-Key"] = self.api_key
        headers.setdefault("Content-Type", "application/json")
        response = self.session.request(method, url, headers=headers, timeout=15, **kwargs)
        if response.status_code >= 400:
            try:
                detail = response.json()
                message = detail.get("message") or detail.get("error") or response.text
            except ValueError:
                message = response.text
            raise ClockifyError(f"{response.status_code}: {message}")
        return response

    def get_user(self) -> Dict[str, Any]:
        return self._request("GET", "/user").json()

    def list_time_entries(
        self,
        workspace_id: str,
        user_id: str,
        start: str,
        end: str,
        page_size: int = 50,
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        page = 1
        while True:
            params = {
                "start": start,
                "end": end,
                "page": page,
                "page-size": page_size,
            }
            response = self._request(
                "GET",
                f"/workspaces/{workspace_id}/user/{user_id}/time-entries",
                params=params,
            )
            batch = response.json() or []
            if not isinstance(batch, list):
                raise ClockifyError(f"Unexpected response for time entries: {batch}")
            entries.extend(batch)
            if len(batch) < page_size:
                break
            page += 1
        return entries

    def create_time_entry(self, workspace_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._request(
            "POST",
            f"/workspaces/{workspace_id}/time-entries",
            json=payload,
        )
        return response.json()


def _summarize_existing(entries: List[EntrySnapshot]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for snap in entries:
        summary.append(
            {
                "description": (snap.entry.get("description") or "").strip(),
                "projectId": snap.entry.get("projectId") or "",
                "start": snap.start,
                "durationSeconds": int(snap.duration.total_seconds()),
            }
        )
    return summary


def _is_duplicate(
    candidate: EntrySnapshot,
    existing: List[Dict[str, Any]],
    *,
    start_tolerance_seconds: int = 300,
    duration_tolerance_seconds: int = 120,
) -> bool:
    description = (candidate.entry.get("description") or "").strip()
    project_id = candidate.entry.get("projectId") or ""
    duration_seconds = int(candidate.duration.total_seconds())
    for item in existing:
        if (item["description"], item["projectId"]) != (description, project_id):
            continue
        start_diff = abs((item["start"] - candidate.start).total_seconds())
        duration_diff = abs(item["durationSeconds"] - duration_seconds)
        if start_diff <= start_tolerance_seconds and duration_diff <= duration_tolerance_seconds:
            return True
    return False


def _select_source_week(
    snapshots: List[EntrySnapshot],
    target_week_start: dt.datetime,
    weeks_back: int,
) -> Tuple[List[EntrySnapshot], Optional[dt.datetime]]:
    for offset in range(1, weeks_back + 1):
        candidate_start = target_week_start - dt.timedelta(days=7 * offset)
        candidate_end = candidate_start + dt.timedelta(days=7)
        candidates = [snap for snap in snapshots if candidate_start <= snap.start < candidate_end]
        if candidates:
            candidates.sort(key=lambda snap: snap.start)
            return candidates, candidate_start
    return [], None


def generate_week_from_history(
    *,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    target_week_start: Optional[str] = None,
    weeks_back: int = 2,
    dry_run: bool = True,
    allow_existing: bool = False,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    timezone_name: Optional[str] = None,
) -> str:
    tz = _resolve_timezone(timezone_name)
    weeks_back = max(1, int(weeks_back or 1))
    week_start_date = dt.date.fromisoformat(target_week_start) if target_week_start else _next_monday()
    target_week_start_dt = dt.datetime.combine(week_start_date, dt.time.min, tzinfo=tz)
    target_week_end_dt = target_week_start_dt + dt.timedelta(days=7)

    client = ClockifyClient(api_key=api_key, base_url=base_url)
    user_payload = client.get_user()
    resolved_workspace = workspace_id or user_payload.get("activeWorkspace") or user_payload.get("defaultWorkspace")
    resolved_user = user_id or user_payload.get("id")
    if not resolved_workspace or not resolved_user:
        raise ClockifyError("Unable to resolve workspace/user from Clockify profile")

    existing_raw = client.list_time_entries(
        resolved_workspace,
        resolved_user,
        start=_isoformat(target_week_start_dt),
        end=_isoformat(target_week_end_dt),
    )
    existing_snapshots = [
        snap
        for snap in (
            EntrySnapshot(entry=item, start=interval[0], end=interval[1])
            for item in existing_raw
            for interval in [_extract_interval(item, tz)]
            if interval
        )
    ]
    if existing_snapshots and not allow_existing:
        result = {
            "status": "skipped-existing",
            "reason": "Target week already has entries. Set allow_existing=true to continue.",
            "existingCount": len(existing_snapshots),
            "targetWeekStart": week_start_date.isoformat(),
            "workspaceId": resolved_workspace,
            "userId": resolved_user,
        }
        return json.dumps(result, indent=2)

    history_start = target_week_start_dt - dt.timedelta(days=7 * weeks_back)
    history_raw = client.list_time_entries(
        resolved_workspace,
        resolved_user,
        start=_isoformat(history_start),
        end=_isoformat(target_week_start_dt),
    )
    history_snapshots = [
        snap
        for snap in (
            EntrySnapshot(entry=item, start=interval[0], end=interval[1])
            for item in history_raw
            for interval in [_extract_interval(item, tz)]
            if interval
        )
    ]

    source_entries, source_week_start = _select_source_week(history_snapshots, target_week_start_dt, weeks_back)
    if not source_entries or not source_week_start:
        result = {
            "status": "no-template",
            "reason": f"No historical entries found in the last {weeks_back} week(s) to copy.",
            "targetWeekStart": week_start_date.isoformat(),
            "workspaceId": resolved_workspace,
            "userId": resolved_user,
        }
        return json.dumps(result, indent=2)

    shift = target_week_start_dt - source_week_start
    planned_entries: List[EntrySnapshot] = []
    for snap in source_entries:
        shifted_start = snap.start + shift
        shifted_end = snap.end + shift
        if shifted_start >= target_week_end_dt:
            continue
        payload = {
            "start": _isoformat(shifted_start),
            "end": _isoformat(shifted_end),
            "description": snap.entry.get("description") or "",
            "billable": bool(snap.entry.get("billable", False)),
        }
        project_id = snap.entry.get("projectId")
        task_id = snap.entry.get("taskId")
        tag_ids = _extract_tag_ids(snap.entry)
        if project_id:
            payload["projectId"] = project_id
        if task_id:
            payload["taskId"] = task_id
        if tag_ids:
            payload["tagIds"] = tag_ids
        planned_entries.append(
            EntrySnapshot(
                entry={
                    **snap.entry,
                    "payload": payload,
                    "sourceEntryId": snap.entry.get("id"),
                },
                start=shifted_start,
                end=shifted_end,
            )
        )

    existing_summary = _summarize_existing(existing_snapshots)
    created: List[Dict[str, Any]] = []
    skipped_duplicates: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for snap in planned_entries:
        payload = snap.entry.get("payload") or {}
        if _is_duplicate(snap, existing_summary):
            skipped_duplicates.append(
                {
                    "description": payload.get("description", ""),
                    "start": _isoformat(snap.start),
                    "durationMinutes": int(snap.duration.total_seconds() // 60),
                    "reason": "duplicate in target week",
                }
            )
            continue
        if dry_run:
            created.append(
                {
                    "description": payload.get("description", ""),
                    "start": payload.get("start"),
                    "end": payload.get("end"),
                    "projectId": payload.get("projectId"),
                    "taskId": payload.get("taskId"),
                    "tagIds": payload.get("tagIds") or [],
                    "sourceEntryId": payload.get("sourceEntryId") or snap.entry.get("id"),
                }
            )
            continue
        try:
            created_entry = client.create_time_entry(resolved_workspace, payload)
            created.append(
                {
                    "description": payload.get("description", ""),
                    "start": payload.get("start"),
                    "end": payload.get("end"),
                    "projectId": payload.get("projectId"),
                    "taskId": payload.get("taskId"),
                    "tagIds": payload.get("tagIds") or [],
                    "createdEntryId": created_entry.get("id"),
                    "sourceEntryId": payload.get("sourceEntryId") or snap.entry.get("id"),
                }
            )
            existing_summary.append(
                {
                    "description": (payload.get("description") or "").strip(),
                    "projectId": payload.get("projectId") or "",
                    "start": snap.start,
                    "durationSeconds": int(snap.duration.total_seconds()),
                }
            )
        except ClockifyError as exc:
            failures.append(
                {
                    "description": payload.get("description", ""),
                    "start": payload.get("start"),
                    "error": str(exc),
                }
            )

    status = "dry-run" if dry_run else "created"
    result = {
        "status": status,
        "workspaceId": resolved_workspace,
        "userId": resolved_user,
        "targetWeekStart": week_start_date.isoformat(),
        "sourceWeekStart": source_week_start.date().isoformat(),
        "dryRun": dry_run,
        "plannedCount": len(planned_entries),
        "createdCount": len(created),
        "skippedDuplicates": skipped_duplicates,
        "failures": failures,
        "entries": created,
    }
    return json.dumps(result, indent=2)
