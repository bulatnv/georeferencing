#!/usr/bin/env bash
# Загрузка датасета OrthoLoC (NeurIPS 2025, deepscenario/OrthoLoC).
#
# Хостинг — webshare TUM, файлы *.npz по ~16 МБ; сплиты и объёмы (замер
# 2026-08-25): test_inPlace 176 файлов (~2.8 ГБ), test_outPlace 1415 (~23 ГБ),
# val 1489 (~24 ГБ), train 13347 (~210 ГБ). Скрипт докачивающий: уже
# существующие файлы пропускаются, обрыв безопасен — просто перезапустить.
#
#   bash scripts/download_ortholoc.sh test_inPlace [data/OrthoLoC] [потоков=8]
#   bash scripts/download_ortholoc.sh all          # все сплиты по возрастанию
set -u

BASE="https://webshare.cvg.cit.tum.de/g/papers/Dhaouadi/OrthoLoC/full"
SPLIT="${1:?сплит: test_inPlace|test_outPlace|val|train|all}"
OUT="${2:-data/OrthoLoC}"
JOBS="${3:-8}"

fetch_split() {
    local split="$1"
    local dir="$OUT/$split"
    mkdir -p "$dir"
    echo "[$split] составляю список..."
    local names
    names=$(curl -s "$BASE/$split/" | grep -oE 'href="[^"]+\.npz"' \
            | sed 's/href="//;s/"//' | sort -u)
    local total have
    total=$(echo "$names" | wc -l)
    have=$(ls "$dir" 2>/dev/null | grep -c '\.npz$' || true)
    echo "[$split] всего $total, локально уже $have"
    rm -f "$dir"/*.part
    # атомарно на файл: качаем в .part и переименовываем только при успехе curl
    echo "$names" | while read -r n; do
        [ -s "$dir/$n" ] || echo "$n"
    done | xargs -P "$JOBS" -I{} -r bash -c \
        'curl -sf --retry 3 --retry-delay 2 -o "$1.part" "$2" && mv "$1.part" "$1"' \
        _ "$dir/{}" "$BASE/$split/{}"
    have=$(ls "$dir" | grep -c '\.npz$')
    echo "[$split] готово: $have/$total"
}

if [ "$SPLIT" = "all" ]; then
    for s in test_inPlace test_outPlace val train; do fetch_split "$s"; done
else
    fetch_split "$SPLIT"
fi
