#!/usr/bin/env bash
# Emit one line whenever the remaining-check count of a watched PR changes.
# Args: repo:pr ... (repo is a fastlmm repo name or owner/repo)   Exits when all watched PRs have no pending checks.
declare -A prev
while true; do
  all_done=1
  for spec in "$@"; do
    repo=${spec%%:*}; pr=${spec##*:}
    json=$(gh pr view "$pr" -R "$( [[ $repo == */* ]] && echo "$repo" || echo "fastlmm/$repo" )" --json statusCheckRollup 2>/dev/null) || { all_done=0; continue; }
    line=$(jq -r '
      [.statusCheckRollup[] | {n:(.name // .context), c:((.conclusion // .state // "") | ascii_upcase)}] as $a
      | ($a | length) as $tot
      | ([$a[] | select(.c=="" or .c=="PENDING" or .c=="EXPECTED")]) as $pend
      | ([$a[] | select(.c=="FAILURE" or .c=="CANCELLED" or .c=="TIMED_OUT" or .c=="ERROR" or .c=="ACTION_REQUIRED")]) as $fail
      | "\($pend|length)|\($tot)|\($fail|length)|\([$pend[]|.n]|join(", "))|\([$fail[]|.n]|join(", "))"' <<<"$json")
    IFS='|' read -r npend tot nfail pendnames failnames <<<"$line"
    [[ "$tot" == 0 ]] && { all_done=0; continue; }
    key="$npend/$nfail"
    if [[ "${prev[$spec]}" != "$key" ]]; then
      old=${prev[$spec]%%/*}
      msg="$repo #$pr: ${old:+$old → }$npend remaining of $tot"
      [[ "$nfail" != 0 ]] && msg="$msg | FAILED($nfail): $failnames"
      if [[ "$npend" == 0 ]]; then msg="$msg | DONE"; elif (( npend <= 6 )); then msg="$msg | waiting on: $pendnames"; fi
      echo "$msg"
      prev[$spec]=$key
    fi
    [[ "$npend" != 0 ]] && all_done=0
  done
  [[ $all_done == 1 ]] && { echo "All watched PRs finished."; exit 0; }
  sleep 45
done
