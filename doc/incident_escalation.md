# Incident Escalation Workflow

## Severity Levels

- **SEV1 — Critical.** A production system is unavailable to most users
  OR there is confirmed data loss OR a security breach is in progress.
  Page the on-call engineer immediately and notify the on-call manager
  within 5 minutes.
- **SEV2 — Major.** A production system is degraded for many users, or
  a critical workflow is broken for a subset of users. Page the on-call
  engineer; manager notification is encouraged but not required.
- **SEV3 — Minor.** A non-critical issue, a single user affected, or a
  cosmetic problem. File a ticket; no paging.
- **SEV4 — Informational.** Tracking only. No on-call action required.

## Who to Page

The on-call rota for each team is published in the on-call directory.
Page through the incident-management tool — never via direct message,
because direct messages are not auditable and on-call may have phone
notifications routed through the tool only.

If you cannot reach the primary on-call within 10 minutes for a SEV1
or SEV2, page the secondary. If neither responds within 20 minutes,
escalate to the engineering manager on the rota.

## During an Incident

- **Open an incident channel** in the chat platform within 5 minutes
  of declaration. Name pattern: `#inc-YYYYMMDD-short-description`.
- **Assign roles:** Incident Commander (drives), Communications Lead
  (talks to stakeholders), Scribe (writes the timeline). One person
  may hold two roles in a small team.
- **Status updates** every 30 minutes, or every 15 minutes for SEV1,
  even if the update is "no change yet".
- **Customer communications** are owned by the Communications Lead in
  consultation with the on-call manager. Engineers do not post
  publicly without that sign-off.

## After an Incident

- **Resolve and close** the incident in the tool when the underlying
  issue is fixed AND any temporary mitigations have been removed or
  ticketed for follow-up.
- **Postmortem within 5 working days** for any SEV1 or SEV2. The
  template lives in the engineering wiki. Postmortems are blameless;
  the goal is to surface systemic causes, not assign individual fault.
- **Action items** from the postmortem are tracked in the team's
  ticketing system with a clear owner and a deadline.

## When in Doubt

If you are unsure of the severity, escalate higher rather than lower.
A SEV2 that turns out to be a SEV3 is a small annoyance; a SEV3 that
turns out to be a SEV1 is a postmortem item.
