#!/usr/bin/env bash
# Emit a line whenever a workflow run's remaining-job count, failures, or status change.
# Args: owner/repo:run_id ...   (owner defaults to fastlmm if omitted)
# Exits when every run has completed. Flags runs waiting for environment approval.
declare -A prev
while true; do
  all_done=1
  for spec in "$@"; do
    repo=${spec%%:*}; id=${spec##*:}
    [[ "$repo" == */* ]] || repo="fastlmm/$repo"
    j=$(gh run view "$id" -R "$repo" --json status,conclusion,jobs 2>/dev/null) || { all_done=0; continue; }
    line=$(jq -r '
      ([.jobs[]|select(.status!="completed")]) as $p
      | ([.jobs[]|select(.conclusion=="failure" or .conclusion=="cancelled" or .conclusion=="timed_out")]) as $f
      | "\($p|length)|\(.jobs|length)|\($f|length)|\([$p[]|.name]|join(", "))|\([$f[]|.name]|join(", "))|\(.status)|\(.conclusion)"' <<<"$j")
    IFS='|' read -r np tot nf pn fn st co <<<"$line"
    key="$np/$nf/$st"
    if [[ "${prev[$spec]}" != "$key" ]]; then
      old=${prev[$spec]%%/*}
      m="${repo#fastlmm/} run $id: ${old:+$old → }$np remaining of $tot (status: $st)"
      [[ "$nf" != 0 ]] && m="$m | FAILED($nf): $fn"
      if [[ "$st" == completed ]]; then m="$m | DONE ($co)"
      elif [[ "$st" == waiting ]]; then m="$m | WAITING FOR APPROVAL"
      elif (( np > 0 && np <= 6 )); then m="$m | waiting on: $pn"; fi
      echo "$m"
      prev[$spec]=$key
    fi
    [[ "$st" != completed ]] && all_done=0
  done
  [[ $all_done == 1 ]] && { echo "All runs finished."; exit 0; }
  sleep 45
done
