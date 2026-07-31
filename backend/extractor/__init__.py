"""Extraction service — a second entrypoint into the same codebase.

Same repo and same .env as the API on purpose: extraction settings feed
config_hash, and two processes reading different configuration would make that
hash describe a pipeline that never ran.
"""
