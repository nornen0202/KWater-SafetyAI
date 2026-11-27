"""Map raw official IDs to portable, reversible file names."""

from urllib.parse import quote


def file_stem(sample_id):
    return quote(sample_id, safe="")
