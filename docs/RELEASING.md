# Releasing `aiollm-gateway` to PyPI

This is a click-by-click guide for publishing this package for the first
time. It assumes you have never published a Python package before. Read the
whole thing once before doing anything — the real release step
(`git push origin v0.1.0`) is irreversible.

## 1. What PyPI and TestPyPI are

- **PyPI** (https://pypi.org) is the real Python Package Index. Anything you
  publish there is what `pip install aiollm-gateway` pulls, forever (see
  "Publishing is irreversible" below).
- **TestPyPI** (https://test.pypi.org) is a separate, throwaway instance of
  the same software, meant for rehearsing an upload without touching the
  real index. It has its own accounts, its own project namespace, and it is
  occasionally wiped. Nothing you do there affects the real `aiollm-gateway`
  project on PyPI.

Always rehearse on TestPyPI first. It catches problems (bad metadata, a
broken workflow, a typo'd environment name) before they become permanent on
the real index.

## 2. How this repo publishes: Trusted Publishing (no API tokens)

`.github/workflows/release.yml` publishes using PyPI's **Trusted
Publishing**: GitHub Actions proves its identity to PyPI directly via OIDC
(an ephemeral, cryptographically signed token GitHub issues to the workflow
run). Nothing is copied, generated, or stored — **you never create a PyPI
API token, and no secret is ever added to this repository.** You only tell
PyPI, once, "trust workflow `release.yml` in this specific GitHub repo."

## 3. Create a PyPI account (and a TestPyPI account)

1. Go to https://pypi.org/account/register/ and create an account (email +
   password; PyPI requires 2FA, so also set up an authenticator app when
   prompted).
2. Separately, go to https://test.pypi.org/account/register/ and create a
   **second, independent** account there. TestPyPI does not share logins
   with PyPI.

## 4. Add a "pending publisher" on PyPI

Because `aiollm-gateway` doesn't exist as a project on PyPI yet, you register
the trust relationship *before* the project exists, via a "pending
publisher."

1. Log in at https://pypi.org.
2. Go to https://pypi.org/manage/account/publishing/ (Account settings →
   Publishing, or just visit that URL directly).
3. Scroll to "Add a new pending publisher" and fill in exactly:

   | Field | Value |
   |---|---|
   | PyPI project name | `aiollm-gateway` |
   | Owner | `Yashwanth-Nallapuneni` |
   | Repository name | `llm-gateway` |
   | Workflow filename | `release.yml` |
   | Environment name | `pypi` |

   The environment name must match what `release.yml` actually declares for
   the `publish-pypi` job (currently `pypi` — check the `environment: name:`
   line in the workflow if this doc and the file ever disagree; the workflow
   is the source of truth).

4. Click "Add". You'll see it listed under "Pending publishers."

**Warning: this does *not* reserve the name `aiollm-gateway`.** A pending
publisher is only a promise — "when a project by this name gets published by
this exact repo/workflow, trust it automatically." Anyone else can still
register the plain name `aiollm-gateway` on PyPI in the meantime, and if
they do, your pending publisher becomes useless and you'd need to pick a
different name. **Publish promptly after adding the pending publisher** —
don't leave a long gap between this step and step 7 (the real release).

## 5. Repeat on TestPyPI (optional but recommended)

Same steps, on https://test.pypi.org/manage/account/publishing/, with:

| Field | Value |
|---|---|
| PyPI project name | `aiollm-gateway` |
| Owner | `Yashwanth-Nallapuneni` |
| Repository name | `llm-gateway` |
| Workflow filename | `release.yml` |
| Environment name | `testpypi` |

This is what lets you run the TestPyPI rehearsal in step 6.

## 6. Create the two GitHub Environments

GitHub's Trusted Publishing integration requires the workflow to run inside
a named "environment" that matches what you typed into PyPI.

1. On GitHub, go to this repo → **Settings → Environments**.
2. Click **New environment**, name it exactly `pypi`, click **Configure
   environment** (no protection rules are required, though you can add
   "required reviewers" here later for extra safety — it just means a human
   has to click approve before the real publish job runs).
3. Repeat for an environment named exactly `testpypi`.

## 7. Rehearse: publish to TestPyPI

1. On GitHub, go to the **Actions** tab → select the **Release** workflow in
   the left sidebar.
2. Click **Run workflow** (the manual `workflow_dispatch` trigger), leave
   branch as `main`, click the green **Run workflow** button.
3. Watch the run. It builds the package, checks it with `twine check`, then
   (because this was a manual trigger, not a tag push) runs the
   `publish-testpypi` job, which uploads to `https://test.pypi.org/legacy/`.
4. When it succeeds, verify the upload actually installs. In a **fresh**
   virtual environment (not this repo):

   ```bash
   python3 -m venv /tmp/testpypi-check
   source /tmp/testpypi-check/bin/activate
   pip install -i https://test.pypi.org/simple/ aiollm-gateway
   python -c "import llm_gateway; print(llm_gateway.__version__)"
   deactivate
   rm -rf /tmp/testpypi-check
   ```

   **Caveat:** this package has no required runtime dependencies, so a plain
   install should work fine. But if you ever add a dependency, note that
   TestPyPI only knows about *other* packages that have also been uploaded
   to TestPyPI — it can fail to resolve a dependency that exists on the real
   PyPI but was never pushed to TestPyPI. That's a TestPyPI quirk, not a
   sign anything is wrong with this package.

## 8. The real release

Once the TestPyPI rehearsal looks right (workflow went green, install
worked):

```bash
git tag v0.1.0
git push origin v0.1.0
```

Pushing a tag matching `v*` triggers `release.yml`'s `publish-pypi` job,
which uploads the build to the real PyPI using the `pypi` environment and
its Trusted Publisher entry. Watch the Actions tab the same way as step 7.

Once it succeeds, the project page is live at
https://pypi.org/project/aiollm-gateway/ and `pip install aiollm-gateway`
works for everyone.

## 9. Publishing is irreversible — read this before you tag

- **A version number can never be reused.** Once `0.1.0` is uploaded to
  PyPI, you can never upload a different `0.1.0` again — even if you delete
  it. Your next release, bug fix or not, must bump the version (`0.1.1`,
  `0.2.0`, ...) in `pyproject.toml` before tagging.
- **A deleted project name cannot be re-registered by you or anyone else.**
  If you ever delete the entire `aiollm-gateway` project from PyPI, that
  name is gone for good — PyPI does not let it be re-claimed, by you or
  anyone else, to prevent supply-chain attacks where someone re-registers a
  formerly-trusted name.
- Because of both of the above, **do not delete releases or projects to fix
  mistakes.** Use yanking instead (below), or just ship a new version.

## 10. Yanking a bad release (not deletion)

If you publish a version that's broken, "yank" it instead of deleting it:

1. Log in to PyPI, go to https://pypi.org/manage/project/aiollm-gateway/releases/.
2. Find the bad version, click the "..." options menu next to it, choose
   **Yank release**, and give a short reason (shown to users).

Yanking means: the file stays on PyPI and anyone who already pinned that
exact version (or has it cached/installed) is unaffected, but `pip install
aiollm-gateway` (with no version pin) will skip it and a fresh
`pip install aiollm-gateway==0.1.0` will warn loudly before installing it.
It is reversible (you can un-yank), unlike deletion, and it does not free up
the version number for reuse.
