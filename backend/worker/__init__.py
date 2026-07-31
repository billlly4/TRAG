"""Ingestion worker — the third process.

Holds no service_role key and no user credentials. Its whole database surface is
the four SECURITY DEFINER functions in 0008_ingest_queue.sql.
"""
