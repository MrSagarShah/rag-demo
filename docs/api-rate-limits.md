# API Rate Limits

The Nimbus API uses token-bucket rate limiting. Limits are applied per API key
and reset on a rolling basis.

## Limits by plan

- **Free tier**: 60 requests per minute and 10,000 requests per day.
- **Pro tier**: 600 requests per minute and 500,000 requests per day.
- **Enterprise tier**: custom limits negotiated per contract.

## Rate limit headers

Every response includes headers describing your current budget:

- `X-RateLimit-Limit` — your ceiling for the current window.
- `X-RateLimit-Remaining` — requests left in the current window.
- `X-RateLimit-Reset` — Unix timestamp when the window resets.

## Handling 429 responses

When you exceed a limit the API returns HTTP `429 Too Many Requests` with a
`Retry-After` header indicating how many seconds to wait. Clients should honor
`Retry-After` and use exponential backoff with jitter for retries.

## Bulk endpoints

Bulk endpoints (such as `/v1/notes/batch`) count each item in the batch as a
separate request against your limit. Requesting a higher limit is possible for
Pro and Enterprise customers by contacting support.
