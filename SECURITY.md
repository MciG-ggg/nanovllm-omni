# Security

## Reporting a vulnerability

If you discover a security issue in `nanovllm-omni`, please report it
privately rather than opening a public issue. Email the maintainers at
the address listed on the GitHub profile
([github.com/MciG-ggg](https://github.com/MciG-ggg)) and include:

- A short description of the issue and its impact.
- Reproduction steps (PoC code, sample weights, sample prompts).
- The commit / tag / version range affected.

You can also use GitHub's private vulnerability reporting: go to the
[Security tab](https://github.com/MciG-ggg/nanovllm-omni/security) of
the repository and click "Report a vulnerability".

## Scope

`nanovllm-omni` runs model weights and untrusted prompts in a single
Python process. Treat prompts as untrusted input; the runtime does not
sandbox HF `trust_remote_code=True` model files. When loading
`trust_remote_code` weights from an untrusted source, prefer the
offline bundle pattern shown in
`examples/offline_inference/minimind_o/README.md` and verify checksums
out-of-band.

## Disclosure timeline

We aim to acknowledge new reports within 7 days and to publish a fix
or a workaround within 30 days of confirmation. Critical issues may
move faster.
