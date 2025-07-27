from cachetools import TTLCache

# Initialize a Time-To-Live (TTL) cache.
# maxsize: The max number of documents to keep in memory.
# ttl: The time in seconds to keep a cached item (e.g., 3600 = 1 hour).
document_cache = TTLCache(maxsize=100, ttl=3600)