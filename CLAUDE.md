# alef — model development & testing

@../sefer/overview.md
@../sefer/conventions.md

Where algorithms are prototyped and validated **on simulated data first**, against
navi's model primitives (`from lib.trading …`, `from lib.db …`). Matured code
graduates into navi. Reference docs live in `sefer/alef/`.

## Environment & commands

- pyenv env `alef-3.11.2`; deps are pip-compiled (`requirements.in` → `.txt`,
  includes `-e ../navi`).
- Development is notebook-driven (`notebooks/random_processes/…`); strategies live
  in `apps/`.
