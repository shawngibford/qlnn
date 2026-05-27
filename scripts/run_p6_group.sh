#!/usr/bin/env bash
#
# run_p6_group.sh -- Staggered P6 system-group runner with explicit
# user go/no-go between systems.
#
# Closes Gap G7 from P6_LAUNCH_PLAN.md. The wrapper:
#   * pre-flights (data dir, venv python, integrity gate)
#   * runs the systems of a named group serially
#   * prompts the user between systems (default-N abort)
#   * tees per-system stdout+stderr to results/p6_<group>/<system>.log
#
# Usage:
#   ./scripts/run_p6_group.sh GROUP_NAME
#   ./scripts/run_p6_group.sh --list
#   ./scripts/run_p6_group.sh --dry-run GROUP_NAME
#
# Exit codes:
#   0  all systems in group ran cleanly
#   1  a system failed
#   2  user aborted between systems
#   3  pre-flight safety check failed
#
# Hard constraints (carried forward from P6_LAUNCH_PLAN.md sec.5):
#   * never `rm` the `data` symlink/dir while jobs run
#   * runs are NOT committed; user controls all git operations
#
set -o pipefail
# Note: we intentionally do NOT `set -u`. macOS ships bash 3.2, where
# empty-array expansions under nounset cause spurious "unbound variable"
# aborts. Each variable is explicitly initialized below.

# ---------------------------------------------------------------------------
# Resolve repo root from this script's location so cwd doesn't matter.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Group registry.
#
# Each group is a list of "system:runner-tag" entries. The runner-tag
# selects which per-phase Python script to dispatch; see _runner_for().
#
# Estimated wall-clocks are author estimates from P6_LAUNCH_PLAN.md and
# are advisory only.
# ---------------------------------------------------------------------------
GROUP_NAMES=(smoke cheap_smooth kuramoto_kdv)

group_systems() {
    case "$1" in
        smoke)
            # M1: single cell, cheapest path (forecaster, LV, seed 0).
            echo "lotka_volterra:forecaster_smoke"
            ;;
        cheap_smooth)
            # M2: solver + forecaster on the two smoothest ODE systems.
            echo "lotka_volterra:solver_h1"
            echo "van_der_pol:solver_h1"
            ;;
        kuramoto_kdv)
            # M3: deferred-completion group (kuramoto + KdV solver H1).
            # Runner is being scaffolded in parallel by M0-G8.
            echo "kuramoto:kuramoto_kdv_h1"
            echo "kdv:kuramoto_kdv_h1"
            ;;
        *)
            return 1
            ;;
    esac
}

group_walltime_estimate() {
    case "$1" in
        smoke)        echo "~10 min" ;;
        cheap_smooth) echo "~30 min" ;;
        kuramoto_kdv) echo "~7-8 hr per cell (~14-16 hr group)" ;;
        *)            echo "unknown" ;;
    esac
}

# ---------------------------------------------------------------------------
# Map a runner-tag to a concrete command (array of argv tokens).
# Echoes one argv token per line so the caller can `mapfile` it.
#
# system arg is $1, runner-tag is $2, out-dir is $3.
# ---------------------------------------------------------------------------
_runner_for() {
    local system="$1"
    local tag="$2"
    local out_dir="$3"

    case "${tag}" in
        forecaster_smoke)
            # M1 smoke: LV + data_reuploading + seed 0 only.
            printf '%s\n' \
                "${PYTHON_BIN}" \
                "scripts/run_p4_forecaster_rollout.py" \
                "--systems" "${system}" \
                "--families" "data_reuploading" \
                "--seeds" "0" \
                "--out" "${out_dir}"
            ;;
        solver_h1)
            # M2 cheap-smooth: dispatches the n=24 solver-H1 runner.
            # That script does not take --systems today; placeholder
            # TODO until G8/G6 extend it. We invoke it as-is so the
            # wrapper does not silently fake a per-system filter.
            printf '%s\n' \
                "${PYTHON_BIN}" \
                "scripts/run_p7_8_h1_n24.py"
            ;;
        kuramoto_kdv_h1)
            # M3 deferred-completion: depends on M0-G8's
            # run_p7_8_h1_kuramoto_kdv.py which may not yet exist.
            local cand="scripts/run_p7_8_h1_kuramoto_kdv.py"
            if [[ -f "${cand}" ]]; then
                printf '%s\n' \
                    "${PYTHON_BIN}" \
                    "${cand}" \
                    "--systems" "${system}"
            else
                printf '%s\n' \
                    "__MISSING_RUNNER__" \
                    "${cand}"
            fi
            ;;
        *)
            printf '%s\n' "__UNKNOWN_TAG__" "${tag}"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Logging helpers (ASCII only; no emoji per spec).
# ---------------------------------------------------------------------------
BAR="===================================================================="

log_info()  { printf '%s\n' "[run_p6_group] $*"; }
log_warn()  { printf '%s\n' "[run_p6_group] WARN: $*" >&2; }
log_error() { printf '%s\n' "[run_p6_group] ERROR: $*" >&2; }

banner() {
    local group="$1" idx="$2" total="$3" system="$4" est="$5" out="$6"
    printf '\n%s\n' "${BAR}"
    printf '  P6 GROUP: %s -- system %d/%d: %s\n' \
        "${group}" "${idx}" "${total}" "${system}"
    printf '  Estimated wall-clock: %s\n' "${est}"
    printf '  Output: %s/\n' "${out}"
    printf '%s\n\n' "${BAR}"
}

# ---------------------------------------------------------------------------
# Argument parsing.
# ---------------------------------------------------------------------------
DRY_RUN=0
GROUP=""
SHOW_LIST=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [--list] [--dry-run] GROUP_NAME

Groups:
$(for g in "${GROUP_NAMES[@]}"; do printf '  %s\n' "${g}"; done)

Options:
  --list      List available groups and their systems, then exit.
  --dry-run   Print what would run for GROUP_NAME without executing.
  -h, --help  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)    SHOW_LIST=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --*)       log_error "unknown flag: $1"; usage >&2; exit 3 ;;
        *)
            if [[ -n "${GROUP}" ]]; then
                log_error "multiple group names given: '${GROUP}' and '$1'"
                exit 3
            fi
            GROUP="$1"
            shift
            ;;
    esac
done

# ---------------------------------------------------------------------------
# --list short-circuits everything (no pre-flight needed).
# ---------------------------------------------------------------------------
if [[ "${SHOW_LIST}" -eq 1 ]]; then
    printf 'Available P6 groups:\n\n'
    for g in "${GROUP_NAMES[@]}"; do
        printf '  %-14s  (%s)\n' "${g}" "$(group_walltime_estimate "${g}")"
        while IFS= read -r entry; do
            sys="${entry%%:*}"
            tag="${entry##*:}"
            printf '      - %-18s [runner: %s]\n' "${sys}" "${tag}"
        done < <(group_systems "${g}")
        printf '\n'
    done
    exit 0
fi

if [[ -z "${GROUP}" ]]; then
    log_error "no group name supplied"
    usage >&2
    exit 3
fi

if ! group_systems "${GROUP}" >/dev/null 2>&1; then
    log_error "unknown group: ${GROUP}"
    log_error "valid groups: ${GROUP_NAMES[*]}"
    exit 3
fi

# ---------------------------------------------------------------------------
# Locate the venv python. Prefer worktree-local, fall back to main repo.
# (Per CLAUDE.md the venv lives at the project root; worktrees usually
# share it.)
# ---------------------------------------------------------------------------
PYTHON_BIN=""
for cand in \
    "${REPO_ROOT}/.venv/bin/python" \
    "/Users/shawngibford/dev/phd/qlnn/.venv/bin/python"
do
    if [[ -x "${cand}" ]]; then
        PYTHON_BIN="${cand}"
        break
    fi
done

# ---------------------------------------------------------------------------
# Pre-flight safety checks. ABORT on any failure (exit 3).
# Skipped under --dry-run so the wrapper is inspectable on broken trees.
# ---------------------------------------------------------------------------
preflight() {
    log_info "pre-flight: data/ directory present"
    if [[ ! -e "${REPO_ROOT}/data" ]]; then
        log_error "data/ is missing (symlink or dir expected at ${REPO_ROOT}/data)"
        log_error "the user-locked rule is: never rm the data symlink while jobs run"
        return 3
    fi

    log_info "pre-flight: .venv/bin/python is executable"
    if [[ -z "${PYTHON_BIN}" ]]; then
        log_error "no usable python found at .venv/bin/python (worktree or main repo)"
        return 3
    fi
    log_info "  using ${PYTHON_BIN}"

    log_info "pre-flight: scripts/verify_paper_integrity.py exits 0"
    local verifier="${REPO_ROOT}/scripts/verify_paper_integrity.py"
    if [[ ! -f "${verifier}" ]]; then
        log_error "verifier missing: ${verifier}"
        return 3
    fi
    if ! "${PYTHON_BIN}" "${verifier}" >/dev/null 2>&1; then
        log_error "verify_paper_integrity.py is NOT exit 0 on the current tree"
        log_error "fix the integrity gate before starting a multi-hour run"
        return 3
    fi

    log_info "pre-flight: all checks passed"
    return 0
}

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log_info "DRY RUN: skipping pre-flight checks"
    # Still resolve PYTHON_BIN to a sentinel so the printed command is honest.
    if [[ -z "${PYTHON_BIN}" ]]; then
        PYTHON_BIN=".venv/bin/python"
    fi
else
    if ! preflight; then
        exit 3
    fi
fi

# ---------------------------------------------------------------------------
# Build the list of (system, tag) for this group.
# bash 3.2 (macOS default) has no `mapfile`, so use a while-read loop.
# ---------------------------------------------------------------------------
ENTRIES=()
while IFS= read -r _line; do
    ENTRIES+=("${_line}")
done < <(group_systems "${GROUP}")
TOTAL=${#ENTRIES[@]}

GROUP_OUT_BASE="results/p6_${GROUP}"
mkdir -p "${GROUP_OUT_BASE}"

EST="$(group_walltime_estimate "${GROUP}")"

log_info "group:    ${GROUP}"
log_info "systems:  ${TOTAL}"
log_info "estimate: ${EST}"
log_info "outputs:  ${GROUP_OUT_BASE}/"
log_info "logs:     ${GROUP_OUT_BASE}/<system>.log"

# ---------------------------------------------------------------------------
# Run loop with explicit between-system go/no-go.
# ---------------------------------------------------------------------------
fmt_elapsed() {
    local s="$1"
    printf '%02d:%02d' $((s / 60)) $((s % 60))
}

partial_path=""
exit_code=0

for idx in "${!ENTRIES[@]}"; do
    entry="${ENTRIES[$idx]}"
    system="${entry%%:*}"
    tag="${entry##*:}"
    pos=$((idx + 1))

    sys_out="${GROUP_OUT_BASE}/${system}"
    sys_log="${GROUP_OUT_BASE}/${system}.log"
    mkdir -p "${sys_out}"

    banner "${GROUP}" "${pos}" "${TOTAL}" "${system}" "${EST}" "${sys_out}"

    # Materialize the runner argv for this (system, tag).
    # bash 3.2-compatible alternative to `mapfile`.
    CMD=()
    while IFS= read -r _tok; do
        CMD+=("${_tok}")
    done < <(_runner_for "${system}" "${tag}" "${sys_out}")

    if [[ "${CMD[0]}" == "__MISSING_RUNNER__" ]]; then
        log_warn "TODO: runner ${CMD[1]} does not exist yet"
        log_warn "  (this is expected if M0-G8 has not landed; aborting"
        log_warn "   rather than silently invoking the wrong script)"
        partial_path="${GROUP_OUT_BASE}"
        exit_code=1
        break
    fi
    if [[ "${CMD[0]}" == "__UNKNOWN_TAG__" ]]; then
        log_error "unknown runner tag in group registry: ${CMD[1]}"
        exit_code=3
        break
    fi

    log_info "command: ${CMD[*]}"
    log_info "log:     ${sys_log}"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log_info "(dry-run) skipping execution"
        continue
    fi

    started=$(date +%s)
    # Tee combined stdout+stderr to per-system log; preserve runner exit
    # via PIPESTATUS (pipefail keeps the failure observable too).
    "${CMD[@]}" 2>&1 | tee "${sys_log}"
    rc=${PIPESTATUS[0]}
    finished=$(date +%s)
    elapsed=$((finished - started))

    if [[ "${rc}" -ne 0 ]]; then
        log_error "system ${system} FAILED (exit ${rc}) after $(fmt_elapsed "${elapsed}")"
        log_error "partial results: ${GROUP_OUT_BASE}/"
        log_error "log: ${sys_log}"
        partial_path="${GROUP_OUT_BASE}"
        exit_code=1
        break
    fi

    log_info "system ${system} completed in $(fmt_elapsed "${elapsed}")"

    # No prompt after the final system.
    if [[ "${pos}" -lt "${TOTAL}" ]]; then
        next_system="${ENTRIES[$((idx + 1))]%%:*}"
        # Default-N: empty answer aborts. Read from controlling tty so
        # the prompt still works when the script's stdout is teed.
        printf '\nSystem %s completed in %s. Continue to %s? [y/N] ' \
            "${system}" "$(fmt_elapsed "${elapsed}")" "${next_system}"
        if [[ -r /dev/tty ]]; then
            read -r answer </dev/tty || answer=""
        else
            read -r answer || answer=""
        fi
        case "${answer}" in
            y|Y|yes|YES)
                log_info "user approved continuation to ${next_system}"
                ;;
            *)
                log_warn "user aborted between systems"
                log_warn "partial results path: ${GROUP_OUT_BASE}/"
                log_warn "to resume, re-run with the remaining systems"
                partial_path="${GROUP_OUT_BASE}"
                exit_code=2
                break
                ;;
        esac
    fi
done

if [[ "${exit_code}" -eq 0 && "${DRY_RUN}" -eq 0 ]]; then
    log_info "group ${GROUP} complete; all ${TOTAL} systems ran cleanly"
    log_info "results under ${GROUP_OUT_BASE}/"
fi

exit "${exit_code}"
