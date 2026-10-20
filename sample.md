# Redis Field Guide

Redis is an in-memory data store. It keeps data in RAM, so reads and writes usually finish in under a millisecond. Redis supports strings, hashes, lists, sets and sorted sets. Many teams use it as a cache in front of a slower database.

## Installation

You can install Redis with a package manager or run it in a container. The official Docker image is the fastest way to try it locally.

### Requirements

| Field   | Value  | Status |
|---------|--------|--------|
| Memory  | 512 MB | Ready  |
| Port    | 6379   | Ready  |
| OS      | Linux  | Ready  |

```bash
docker run -d --name redis -p 6379:6379 redis:7
redis-cli ping   # -> PONG
```

## Persistence

Redis can persist data to disk in two ways. RDB snapshots write the whole dataset at intervals, while the append-only file logs every write. Snapshots are compact and fast to restore. The append-only file loses less data after a crash, but it grows larger over time.

## Eviction Policies

When memory fills up, Redis must decide what to remove. The allkeys-lru policy evicts the least recently used keys across the whole keyspace. The volatile-ttl policy only evicts keys that have an expiry, starting with the ones closest to expiring. Choosing the wrong policy can silently delete data you expected to keep.

## Team Notes

The quarterly offsite will be held in Montreal in March. Travel bookings must be submitted to the operations team two weeks in advance. Lunch on the first day is provided by the venue, and dinner is at a restaurant in the Old Port.

However, the budget for the offsite was reduced this year. Each team has a spending cap of four thousand dollars. Expenses above the cap need approval from a director.
