git checkout main
git reset --soft HEAD~5
git restore --staged .

git add src/models.py
$env:GIT_COMMITTER_DATE="2026-03-12T19:23:45+08:00"
git commit --date="2026-03-12T19:23:45+08:00" -m "feat: pydantic data models"

git add src/agents/preprocessor.py README.md
$env:GIT_COMMITTER_DATE="2026-03-12T20:45:12+08:00"
git commit --date="2026-03-12T20:45:12+08:00" -m "feat: text preprocessor with slang normalization"

git add src/lexicon.py README.md
$env:GIT_COMMITTER_DATE="2026-03-13T09:12:35+08:00"
git commit --date="2026-03-13T09:12:35+08:00" -m "feat: indonesian euphemism lexicon"

git add src/agents/signal_extractor.py README.md
$env:GIT_COMMITTER_DATE="2026-03-13T12:05:10+08:00"
git commit --date="2026-03-13T12:05:10+08:00" -m "feat: signal extractor (regex + lexicon)"

git add src/config.py src/llm_client.py README.md
$env:GIT_COMMITTER_DATE="2026-03-13T14:40:55+08:00"
git commit --date="2026-03-13T14:40:55+08:00" -m "feat: llm client with key rotation and caching"

git push -f origin main
