# AEY Inc. TAMU Datathon entries

This repo holds team AEY Inc.'s entries for two challenges. Each challenge has its own folder on `main`, and each one is submitted differently.

| Challenge | Folder | What gets submitted | What the organizers run |
|---|---|---|---|
| Heads-up Short Deck 5-card PLO poker bot | [`FiWBot/`](FiWBot/) | This repo's URL and a full commit SHA from the `submission` branch | A container built from exactly that commit |
| House Edge next-spin payout forecasting | [`house-edge/`](house-edge/) | A forecast CSV and a code ZIP, uploaded through the round's Google Form | The code in the ZIP, through its `reproduce.json` |

## Branches

| Branch | Contents | Use it for |
|---|---|---|
| `main` | Both challenge folders | Working on and reading either project |
| `submission` | The contents of `FiWBot/` at the top level, with nothing from `house-edge/` | The poker bot's Commit SHA |

The poker build needs `agent.py` and `requirements.txt` at the top level of the submitted commit. On `main` they are inside `FiWBot/`, so **a commit SHA from `main` will not build**. Always submit a SHA from `submission`.

The poker build only looks at the commit it is given, so it never sees the other challenge. House Edge doesn't use this repo for submission at all.

## FiWBot

Everything for the bot is in `FiWBot/`. See [`FiWBot/README.md`](FiWBot/README.md) for how it works.

To test it, run these from inside the folder:

```sh
python -m pytest -q        # unit tests
./tools/check_build.sh     # builds and smoke-tests it like the tournament
```

After changing the bot, regenerate the top-level commit and move `submission` to it, from the repo root:

```sh
SHA=$(git subtree split --prefix=FiWBot main)
git push --force origin "$SHA:refs/heads/submission"
echo "$SHA"                # the full SHA for the form
```

## House Edge

The model is in `house-edge/entry/`. See [`house-edge/README.md`](house-edge/README.md) for the approach and how to run it.

Organizer data and generated forecasts are deliberately not in this repo. To run the code, extract the participant kit to `house-edge/kit/`. The folders `kit/`, `dl/`, `runs/` and `submissions/`, and the CSVs in `work/`, are git-ignored so they stay local.

To package an entry, run this from `house-edge/kit/`:

```sh
python3 build_entry.py --source ../entry --out ../entry.zip
```

Upload that ZIP together with the matching test forecast CSV.
