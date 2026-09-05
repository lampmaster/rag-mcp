# Service Authentication and Secrets

This document describes how internal services authenticate to each other and
how the secret material is stored and rotated.

## Bearer credentials

Every call to an internal HTTP API must present a credential in the
`Authorization` header using the Bearer scheme:

```
Authorization: Bearer <token>
```

The value is the service token issued by the identity service. Requests without
the header are rejected with `401 Unauthorized`; requests with an expired
credential are rejected with `401` and the `WWW-Authenticate` header explains
which scope was missing.

## AUTH_TOKEN

Local processes and CI jobs read their credential from the AUTH_TOKEN
environment variable. The value is a short lived JSON Web Token with a lifetime
of one hour, so it must not be committed to the repository or baked into an
image. In CI the variable is injected by the pipeline; on a laptop it is
produced by:

```bash
eval "$(platform auth login --print-env)"
```

If AUTH_TOKEN is missing the client library raises `MissingCredentialError`
before any network call is attempted.

## JWT_SECRET

Tokens are signed with HS256 using the shared signing key held in JWT_SECRET.
The key lives in the secret manager and is mounted into the pod at start up. It
is rotated every thirty days; both the current and the previous key are accepted
during a twenty four hour overlap window so that in-flight tokens stay valid.
Never log JWT_SECRET, and never reuse it between the staging and production
environments.

## Scopes and least privilege

A token carries the scopes granted to the requesting service. Grant the
narrowest scope that still lets the job finish - a batch job that only reads
reports should not receive a write scope. Scope changes take effect on the next
token issue, not immediately.

## Incident response

If you suspect a credential leaked, revoke it first and investigate afterwards.
Revocation is instant, while waiting for natural expiry leaves an attacker a
full hour of access. Report the incident in the security channel with the
affected service name and the approximate time window.

## Rotating a service credential

Rotation is a three step process. First issue the new credential and add it to
the secret manager under the same name with a new version. Second, restart the
consumers so that they pick the new version up; a rolling restart is enough,
there is no need for a maintenance window. Third, disable the old version once
the dashboards show no traffic authenticating with it. Skipping the second step
is what turns a routine rotation into an outage.

## Machine to machine flow

A service asks the identity service for a token by presenting its workload
identity. No password is involved: the platform trusts the identity supplied by
the orchestrator. The identity service answers with a signed token and a
lifetime; the client library caches it in memory and refreshes it when a third
of the lifetime is left. Tokens are never written to disk.

## Transport security

All internal traffic uses TLS 1.3 with certificates issued by the internal
certificate authority. Certificates are rotated automatically every ninety days
by the sidecar. A service that pins a certificate fingerprint will break at the
next rotation - pin the issuing authority instead, or verify the hostname only.

## Handling secrets in configuration

Secrets are mounted as files and read at start up; environment variables are
acceptable for local development but are visible in process listings and crash
dumps. Never interpolate a secret into a log line, an error message or a URL
query string. Values that look like credentials are redacted by the log
pipeline, but the redaction is a safety net and not a guarantee.

## Audit logging

Every issued and revoked credential is recorded in the audit log together with
the requesting workload, the granted scopes and the source address. The log is
append only and retained for one year. During an investigation, start from the
audit log rather than from application logs: it answers who was allowed to do
what, while application logs only show what was attempted.

## Common failures

A `401` with `invalid signature` means the verifying service holds a different
signing key than the issuer - typically a pod that was not restarted after a
rotation. A `403` means the credential is valid but the scope is missing; the
fix is a scope grant, not a new credential. A credential that works from a
laptop but not from a pod usually indicates that the environment variable was
exported in the shell but never added to the deployment manifest.

## Review checklist

Before shipping a service that talks to another internal API, confirm that the
credential is read from configuration rather than hard coded, that failures to
obtain it are fatal at start up rather than silently ignored, that the granted
scopes are the narrowest that work, and that no test fixture contains a real
credential.
