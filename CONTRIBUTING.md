# Contributing

Contributions are welcome. Most pull requests will be accepted as long as they follow the rules below.

What matters most is improving functionality, usability, and user experience. This is an open-source AI project -- keep the barrier low and focus on making things better.

## Language

Write documentation, comments, and commit messages in whatever language you are most comfortable with. This is an AI-native project -- translation is a solved problem. Technical clarity in your own language is better than awkward English.

## Rules

1. **Keep it simple.** Small, focused PRs are preferred over large sweeping changes.
2. **Follow existing patterns.** Match the code style, naming conventions, and project structure already in place.
3. **No security issues.** All PRs are reviewed for security concerns. PRs that introduce vulnerabilities (command injection, path traversal, credential exposure, etc.) will be rejected without exception.
4. **No secret files.** Never commit `.env`, API keys, tokens, or any credentials.
5. **Test before submitting.** Make sure the application runs and your changes work as expected.
6. **One concern per PR.** Bug fix, feature, or refactor -- pick one per pull request.

## Recommended Workflow

This project is developed with [Claude Code](https://claude.ai/claude-code). Using it for contributions is strongly recommended. It understands the full codebase and helps you write consistent code.

```bash
# fork and clone
git clone https://github.com/<you>/video-transcriber.git
cd video-transcriber
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# create a branch
git checkout -b feat/your-feature

# make changes, test, commit, push
python -m src.main
git push origin feat/your-feature

# open a pull request against main
```

## Project Structure

```
src/                 -- Application source code
docs/
  plans/             -- Design documents and implementation plans
  superpowers/       -- Detailed specs for major features
assets/icon/         -- Application icons
build/               -- PyInstaller spec and Inno Setup config
tests/               -- Tests
```

## What Gets Accepted

- Bug fixes
- Performance improvements
- New features that fit the project scope (media transcription)
- UI/UX improvements
- Documentation and translations
- Test coverage improvements

## What Gets Rejected

- Changes that introduce security vulnerabilities
- PRs that include secrets, credentials, or tokens
- Unrelated or out-of-scope features
- Changes that break existing functionality without justification

## CI

When a PR is merged to `main`, the GitHub Actions workflow automatically builds a Windows executable and creates a release. Make sure your changes don't break the build.
