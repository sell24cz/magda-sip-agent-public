# Security

Never report real SIP credentials, API keys, phone numbers, PESEL data, call
recordings, or logs in a public issue.

Use environment variables or ignored local `sip.env` files for secrets. Bind
SIP and RTP only to trusted interfaces and restrict network access with a
firewall or VPN appropriate for your deployment.

This experimental project does not provide a production security guarantee.
