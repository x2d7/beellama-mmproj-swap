# Проверки после обновления до BeeLlama v0.4.9

Ветка: `upgrade/v0.4.9` (база `origin/v0.4.9` = `b7902d18d`, upstream `d327bf1f0`).
Цель: сохранить только `--mmproj-gpu-swap` (+ прогресс логов) и `merge_group`;
control-vector и Muse не переносятся; checkpoint-sidecar заменён штатным механизмом v0.4.9.

## 0. Источник и целостность

- [x] 0.1 Диф против `origin/v0.4.9` содержит только mmproj-swap файлы.
- [x] 0.2 Нет следов cvec (`cvec_mode`, `SET_CVEC`, `/control-vectors`) и Muse-патча.
- [x] 0.3 Нет конфликтных маркеров `<<<<<<<`.
- [x] 0.4 `git status` — только ожидаемые файлы.

## 1. Сборка

- [x] 1.1 `cmake -B build-v049 -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DCMAKE_BUILD_TYPE=Release` без ошибок.
- [x] 1.2 `cmake --build build-v049` без ошибок.
- [x] 1.3 `llama-server --version` → 0.4.9.
- [x] 1.4 В `--help` есть `--mmproj-gpu-swap`, `--mmproj-gpu-swap-reserve-mb`, `--kv-tail-tokens`, `--kvarn-window-chunk`, `--spec-draft-ubatch-size`, `--rs-rollback`.
- [x] 1.5 В `--help`/коде нет `/control-vectors`.

## 2. Автотесты

- [x] 2.1 `ctest -R "kvarn|cuda-fattn-route-policy|save-load-state|llama-context|meta-context-headroom"`.
- [~] 2.2 `pytest tools/server/tests/unit/test_slot_save.py`.
- [ ] 2.3 (опционально) полный `ctest`.

## 3. Базовый запуск (Vireth: IQ4_XS, kvarn6, 98k, mmproj swap)

- [x] 3.1 Сервер стартует, `/health` = ok, VRAM укладывается.
- [x] 3.2 Текстовый запрос: осмысленный ответ, есть usage/timings.
- [ ] 3.3 `GGML_KVARN_DEBUG_ROUTES=1`: native KVarN-route, без materialization fallback.

## 4. mmproj GPU swap

- [x] 4.1 1 картинка: encode → swap → decode, ровно один цикл swap, ответ по содержимому.
- [x] 4.2 3 картинки в одном запросе: один цикл swap на задачу (не три), правильный счёт.
- [x] 4.3 Пик VRAM при vision ≤ 16 ГБ, без OOM.
- [x] 4.4 Без `--mmproj-gpu-swap` vision работает (off-путь не сломан).
- [x] 4.5 ГЛАВНОЕ: контекст text→img→text→img→text→img **без кэша**
      (`cache_prompt=false`): ровно **1** `swapping mmproj to GPU` и **1** `swapping mmproj back to CPU`,
      все картинки кодируются в этом одном цикле (лог `image encoding n/total`), счёт и порядок верные.
- [x] 4.6 Повтор той же раскладки с другим текстовым префиксом (гарантированно свежий промпт): снова 1 цикл.
- [x] 4.7 Модель распознаёт цифры на тестовых картинках (1/2/3) — проверка, что картинки не перепутаны.

## 5. KV cache и качество

- [ ] 5.1 `--kv-tail-tokens 1024` стартует, effective tail в логе.
- [ ] 5.2 Perplexity kvarn6 tail=0 vs tail=1024 на одинаковых `-b/-ub` (ожидаем не хуже).
- [ ] 5.3 `kvarn5/4` (как VirethFast) стартует.

## 6. Slot save/restore на реальном диске (`/mnt/hdd/slot_saves`)

Диск: `/dev/sdb2`, NTFS через FUSE (fuseblk), HDD. Замерено: запись ~91 МБ/с,
чтение ~412 МБ/с. Существующие сейвы: 1.3–2.3 ГБ + `.ckpt` до 1.4 ГБ.

- [x] 6.1 Save создаёт state-файл + checkpoint-приложение (v0.4.9).
- [x] 6.2 Restore проходит, чекпоинты восстанавливаются (`n_checkpoints`/лог).
- [x] 6.3 ГЛАВНОЕ: после restore **нет полного перечитывания контекста**:
      `cached_tokens` ≥ 95% длины промпта, TTFT в разы меньше полного префилла.
- [x] 6.3a Продолжение диалога после restore (промпт + несколько токенов) тоже переиспользует префикс.
- [x] 6.4 Старый slot-файл v0.4.6 отвергается с внятной ошибкой, без краша.
- [x] 6.5 Время save на ~12k токенов: не хуже v0.4.6 (sidecar), МБ/с ≥ ~50.
- [x] 6.6 Время restore (холодный старт сервера) и TTFT после него: TTFT ≪ полного префилла.
      Скорость важна меньше, чем сам факт отсутствия полного перечитывания.
- [x] 6.7 Прогрев/чтение существующего файла 2.3 ГБ: оценить реальную скорость чтения NTFS.
- [x] 6.8 Сравнение с бинарником v0.4.6 на том же промпте (save/restore/TTFT).
- [x] 6.9 mtmd + чекпоинты: сохранить слот с картинками (text→img→text→img), **холодный** restore,
      продолжить диалог: `cached_tokens` ≥ 95% первого промпта и **0** новых mmproj-свапов
      (картинки повторно не кодируются).

## 7. Скорость

- [ ] 7.1 `llama-bench` pp512/pp4096/tg128 на Q3_K_XL, сравнение с бинарником v0.4.6.
- [ ] 7.2 Prefill `-ub 128` vs `-ub 48` (GDN MMA включается при ≥64).
- [ ] 7.3 VirethFast (MTP): стартует, draft accepted > 0, tok/s зафиксирован.

## 8. Логи

- [ ] 8.1 В логах прогонов нет `error|failed|assert|OOM` (кроме ожидаемых негативных тестов).

---

## Результаты прогона (2026-10-29, RTX 5060 Ti 16 GB)

### Сборка и запуск
- `llama-server --version` → `0.4.9-dev (build 12929, commit b7902d18d-dirty)`.
- Сборка `-j12` без ошибок (с `-j16` ловил OOM на шаблонах FA).
- VRAM после загрузки Vireth-подобного профиля (Q3_K_XL, kvarn6, 98k): 14 488 / 16 311 MiB.

### mmproj swap (п. 4)
- `text→img→text→img→text→img`, `cache_prompt=false`: **1** swap in / **1** swap out, все картинки в одном цикле
  (`image encoding n_images = 1/3, 2/3, 3/3`), затем `image decoding 1/3, 2/3, 3/3`.
- Повтор с другим текстовым префиксом: снова 1/1.
- Цифры на картинках распознаны: одиночная «2» → «2»; набор A/B/C (3,1,2) → «3, 1, 2».
- Без `--mmproj-gpu-swap` vision работает, свапов 0.
- `--mmproj-gpu-swap-reserve-mb 2048`: вытеснение 2966 MiB весов ~2 мс, возврат с диска ~8.4 с
  (это цена именно reserve-mb; без него свап только RAM↔VRAM).

### Slot save/restore на /mnt/hdd (п. 6)
| | v0.4.6 (наш sidecar) | v0.4.9 (штатный appendix) |
|---|---:|---:|
| Промпт | 11 850 ток | 11 850 ток |
| Полный префилл | 25 897 мс | 24 782 мс |
| Файл | 467 MiB + 166 MiB `.ckpt` | 658 MiB (один файл) |
| Save | 4.61 с | 4.66 с |
| Restore (чтение+parse) | 6.25 с | 6.45 с |
| TTFT после restore | 1 185 мс | 858 мс |
| **cached_tokens** | **11 776** | **11 776** |
| Продолжение диалога | cached 11 776, 857 мс | cached 11 776, 851 мс |

Вывод: полного перечитывания нет ни там, ни там; v0.4.9 чуть быстрее по TTFT.
Старый файл v0.4.6 отвергается: HTTP 400 «Invalid slot save file», сервер остаётся живым.

### Ограничение: mtmd + restore с KVarN (п. 6.9)
- С KVarN (`kvarn6`) после restore/image-истории `cached_tokens = 0`, TTFT = полный префилл
  (`cache_reason = no_restorable_kvarn_boundary`). Причина: 128-токенная граница KVarN попадает
  внутрь медиа-чанка, а `llama_memory_seq_rm_plan` отказывается резать суффикс с медиа.
- **То же самое на v0.4.6** (проверено тем же сценарием) → не регрессия апгрейда.
- **Работает с обычным кэшем**: `-ctk q8_0 -ctv q8_0`, ctx 16k:
  - 2-й ход: cached 1 181 / 1 204, 1 163 мс;
  - save 669 MB → холодный restore → 3-й ход: cached 1 200 / 1 220, 1 307 мс, ответ верный.
- Рекомендация: для vision-сессий отдельный профиль с `q8_0`/`q6_0` и умеренным ctx
  (98k с q8_0 не влезет в 16 ГБ вместе с IQ4_XS).

### Автотесты (п. 2)
- `ctest -R "kvarn|route-policy|save-load-state|llama-context|meta-context-headroom|std-kv-tail"`:
  **27/28 passed**. Падает `test-save-load-state-tail` (nmse=0.0053) — **падает и на v0.4.6**
  (nmse=0.0559), т.е. предсуществующая проблема fork'а, не апгрейда.
- `pytest test_slot_save.py`: не запускается в этом окружении (нужен HF tiny-model и сеть,
  старт сервера не укладывается в 60 с). Покрыто собственными тестами save/restore выше.

### KVarN + mtmd: как включено переиспользование (новое)

Причина отказа была не в KVarN, а в рекуррентной памяти: гибридная модель (48 GDN-слоёв)
не может откатить хвост без снапшотов (`n_rs_seq = 0` по умолчанию, включался только для
спекулятивного декодирования). KVarN при этом план выдаёт корректно.

Добавлена опция **`--rs-rollback N`** (common/arg.cpp, common/common.cpp):
`n_rs_seq = max(need_n_rs_seq(), N)`. Цена: один снапшот ≈ **103 МБ VRAM** для Qwen3.8-27B
(RS-буфер 103 МБ → 931 МБ при N=8).

Замеры (Q3_K_XL, kvarn6, ctx 24k, `--rs-rollback 8`, RTX 5060 Ti):

| Сценарий | cached | TTFT | swaps |
|---|---:|---:|---:|
| повтор image-промпта (1 картинка в конце) | 1281 / 1282 | 46 мс | — |
| 2-й ход диалога с 2 картинками | 1181 / 1204 | — | — |
| save → **холодный** restore → 3-й ход | 1200 / 1220 | 615 мс | 0 |

Полный префилл тех же 1282 токенов — ~2.6 с, холодный reprocess image-диалога — ~40 с.
VRAM после загрузки с N=8 и ctx 24k: 14 998 / 16 311 MiB.
Без `--rs-rollback` поведение прежнее: cached 0, VRAM 14 488 MiB (регрессии нет).

Рекомендация: для vision-профиля `--ctx-size 24000..32000 --rs-rollback 8`
(на 98k с IQ4_XS +931 МБ не влезает; при 84k — на грани).
