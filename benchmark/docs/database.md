# Database Guide

The platform stores its relational data in PostgreSQL 16. This guide covers
connection settings, pooling and the most common operational problems.

## Connecting

The application reads its connection string from the DATABASE_URL environment
variable:

```
DATABASE_URL=postgresql://api:secret@db.internal:5432/platform
```

The URL must include the schema, the host and the database name. Query
parameters are forwarded to the driver, so `?sslmode=require` can be appended
for environments that enforce TLS. Local development uses the PostgreSQL
container defined in the compose file, which listens on port 5432.

## Connection pooling

Each application worker keeps its own pool of open connections. The size of that
pool is controlled by DB_POOL_SIZE, which defaults to 10. The total number of
connections a deployment opens is therefore `DB_POOL_SIZE * number of workers`,
and PostgreSQL rejects everything above `max_connections`.

When too many clients try to connect at the same time, new requests block until
a slot is released and eventually fail with a pool timeout. The symptom is a
sudden spike of `TimeoutError: QueuePool limit reached` in the API logs while
the database itself looks idle. Lower the number of workers or raise
`max_connections` on the server - raising DB_POOL_SIZE alone only moves the
queue from the application to the database.

## Migrations

Schema changes are applied with Alembic:

```bash
alembic upgrade head
```

Migrations run automatically during deployment, before the new pods receive
traffic. Write migrations so that the previous release keeps working: add
columns as nullable first, backfill in a separate step, and only then add the
constraint.

## Backups and restore

A base backup is taken every night and write-ahead logs are shipped
continuously, which allows point in time recovery. Restoring a single table is
not supported directly; restore into a scratch instance and copy the rows over.
Test the restore procedure once per quarter - a backup that was never restored
is not a backup.

## Slow queries

Statements slower than 500 ms are written to the slow query log. Start any
investigation with `EXPLAIN (ANALYZE, BUFFERS)`. The most frequent cause of a
sudden slowdown is a missing index on a foreign key that was added by a recent
migration.

## Read replicas

Reporting traffic is served by an asynchronous replica. Replication lag is
usually below one second but grows during bulk imports, so a read that must
observe a write made moments earlier has to go to the primary. The application
exposes an explicit `read_from_primary()` context manager for those cases;
routing every read to the primary "just to be safe" is what exhausts the
primary's capacity.

## Transactions and locking

Keep transactions short. A transaction that stays open while the request waits
on an external HTTP call holds its locks for the whole round trip, and every
other statement that touches the same rows queues behind it. Deadlocks are
reported as `deadlock detected` and are almost always caused by two code paths
updating the same two tables in a different order; fix them by agreeing on a
single order rather than by retrying.

## Indexes

Every foreign key needs an index unless the table is tiny and stays tiny.
Create indexes concurrently in production so that writes are not blocked, and
remember that a concurrent build can fail and leave an invalid index behind -
check `pg_index.indisvalid` afterwards and drop what did not finish.

Partial indexes are the cheapest win for tables where queries always filter on
the same flag, for example only the rows that are not yet processed.

## Vacuum and bloat

Autovacuum reclaims the space used by deleted and updated rows. Tables with a
high update rate can still bloat if autovacuum cannot keep up; the symptom is a
table whose size grows while the row count stays flat. Tune the per table
autovacuum thresholds instead of running a manual full vacuum, which takes an
exclusive lock and rewrites the whole table.

## Local development database

The compose stack starts PostgreSQL with a named volume, so the data survives
between sessions. Reset it with `docker compose down -v` when a migration goes
wrong locally. Seed data is loaded by `make seed`, which is idempotent and safe
to run repeatedly.

## Capacity planning

Watch three numbers: the connection count against `max_connections`, the cache
hit ratio, and the replication lag. Connection exhaustion is the failure mode
that hits first in practice, because every new worker multiplies the pool.
Before adding workers to increase throughput, verify that the database still
has connection headroom for them.

## Emergency procedures

If the primary becomes unresponsive, do not restart it blindly - capture
`pg_stat_activity` first, since the query that caused the incident disappears
with the restart. Failover to the standby is a documented, scripted procedure
and takes about ninety seconds; announce it in the incident channel before
starting, because in-flight transactions are lost.
