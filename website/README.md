# DynNav research website

This directory contains the Next.js research landing page. The repository-level
web build combines it with the actual DynNav Researcher frontend under
`/researcher`.

**Status:** Website build is **Implemented** and checked by GitHub Actions. The site is documentation and presentation software; it is not a robotics runtime, safety monitor, ROS2 node, or experimental result.

## Development

```bash
cd website
npm install --no-audit --no-fund
npm run dev
```

## Required checks

```bash
npm run typecheck
npm run build
```

The current CI uses Node 22. Package versions are pinned in
[`package.json`](package.json), the lockfile is committed, and CI uses
`npm ci`. The complete dependency trees are also audited before deployment
builds.

## Combined portal

From the repository root:

```bash
npm --prefix website ci --no-audit --no-fund
npm --prefix apps/web ci --no-audit --no-fund
bash scripts/build_web_portal.sh
python -m http.server 4173 --directory .web-dist
```

The resulting portal serves the research site at `/` and the Researcher
frontend at `/researcher/`. The latter still needs the FastAPI service to run
experiments; static hosting does not simulate or fabricate backend evidence.

## Content policy

- Describe only capabilities present in the repository.
- Label synthetic benchmarks and prototype features explicitly.
- Do not claim ROS2/Nav2 loading, Gazebo runs, hardware tests, formal verification, or safety guarantees without linked evidence.
- Keep media links consistent with [`../assets/`](../assets/README.md) and generated outputs under [`../results/`](../results/README.md).

See the [root README](../README.md) and [`../docs/README.md`](../docs/README.md).
