"""Привязка процесса к производительным ядрам гибридного CPU (Windows).

Зачем. У Alder Lake (наш i7-12700 — 8 P-ядер с гипертредингом плюс 4 E-ядра,
итого 20 логических) планировщик Windows 10 распределяет счётные процессы без
подсказок и охотно кладёт их на энергоэффективные ядра. Для пакетной генерации
это прямая потеря: E-ядро считает то же окно ортоплана заметно дольше, а
воркеров у нас ровно столько, сколько мы задали, — «догнать» отставшего некому.

Ядра определяются не эвристикой «последние четыре — энергоэффективные», а тем,
что сообщает система: ``GetLogicalProcessorInformationEx`` отдаёт для каждого
физического ядра его ``EfficiencyClass``, где больший класс — более
производительное ядро. На не-Windows и при отказе API модуль ничего не делает
и честно об этом говорит — привязка это ускорение, а не условие работы.

Дочерние процессы Windows наследуют маску родителя в момент создания, поэтому
достаточно вызвать `pin_to_performance()` один раз в запускающем скрипте:
`generate.py` и `shift_field.py`, поднятые из пула, унаследуют её сами.

    python open_orto/scripts/cpu_affinity.py        # показать раскладку ядер
"""

from __future__ import annotations

import ctypes
import struct
import sys

RELATION_PROCESSOR_CORE = 0
ERROR_INSUFFICIENT_BUFFER = 122


def _kernel32():
    """kernel32 с объявленными типами.

    Без них ctypes считает возврат ``GetCurrentProcess`` 32-битным, и
    псевдодескриптор процесса (-1) усекается — вызовы молча отказывают.
    """
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.GetCurrentProcess.argtypes = []
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k.SetProcessAffinityMask.restype = ctypes.c_bool
    k.GetProcessAffinityMask.argtypes = [ctypes.c_void_p,
                                         ctypes.POINTER(ctypes.c_size_t),
                                         ctypes.POINTER(ctypes.c_size_t)]
    k.GetProcessAffinityMask.restype = ctypes.c_bool
    k.GetLogicalProcessorInformationEx.argtypes = [ctypes.c_int, ctypes.c_void_p,
                                                   ctypes.POINTER(ctypes.c_ulong)]
    k.GetLogicalProcessorInformationEx.restype = ctypes.c_bool
    return k


def _core_records() -> list[tuple[int, int]]:
    """[(EfficiencyClass, маска логических процессоров), ...] по физическим ядрам.

    Разбор буфера ``SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX`` вручную: записи
    переменной длины, и ctypes-структуру с массивом переменного размера
    описывать дольше, чем прочитать поля по смещениям.
    """
    if not sys.platform.startswith("win"):
        return []
    kernel32 = _kernel32()
    size = ctypes.c_ulong(0)
    kernel32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, None,
                                              ctypes.byref(size))
    if ctypes.get_last_error() not in (0, ERROR_INSUFFICIENT_BUFFER) and size.value == 0:
        return []
    buf = ctypes.create_string_buffer(size.value)
    if not kernel32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf,
                                                     ctypes.byref(size)):
        return []

    data = buf.raw[: size.value]
    out: list[tuple[int, int]] = []
    off = 0
    while off + 8 <= len(data):
        relation, rec_size = struct.unpack_from("<II", data, off)
        if rec_size <= 0 or off + rec_size > len(data):
            break
        if relation == RELATION_PROCESSOR_CORE:
            # PROCESSOR_RELATIONSHIP: Flags(1) EfficiencyClass(1) Reserved(20)
            #                         GroupCount(2) GroupMask[GroupCount]
            eff = data[off + 9]
            group_count = struct.unpack_from("<H", data, off + 30)[0]
            mask = 0
            for g in range(group_count):
                # GROUP_AFFINITY: Mask(ULONG_PTR) Group(2) Reserved(6)
                gm_off = off + 32 + g * 16
                if gm_off + 8 > len(data):
                    break
                mask |= struct.unpack_from("<Q", data, gm_off)[0]
            out.append((eff, mask))
        off += rec_size
    return out


def cores_by_class() -> dict[int, list[int]]:
    """{EfficiencyClass: [номера логических процессоров]} — больший класс быстрее."""
    by: dict[int, list[int]] = {}
    for eff, mask in _core_records():
        cpus = [i for i in range(64) if mask >> i & 1]
        by.setdefault(eff, []).extend(cpus)
    return {k: sorted(v) for k, v in sorted(by.items())}


def performance_mask() -> int:
    """Маска логических процессоров производительных ядер (0, если их не отличить).

    Ноль — не ошибка, а «делить нечего»: на однородном CPU все ядра равны и
    сужать маску незачем.
    """
    by = cores_by_class()
    if len(by) < 2:
        return 0
    top = max(by)
    mask = 0
    for cpu in by[top]:
        mask |= 1 << cpu
    return mask


def pin_to_performance(verbose: bool = True) -> int:
    """Привязать текущий процесс к производительным ядрам; вернуть их число.

    Ноль означает «привязка не применялась» — не гибридный CPU, не Windows
    или отказ API. Вызывающий продолжает работу как обычно.
    """
    mask = performance_mask()
    if not mask:
        if verbose:
            print("привязка к ядрам: не применяется (ядра неразличимы)")
        return 0
    kernel32 = _kernel32()
    handle = kernel32.GetCurrentProcess()
    if not kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask)):
        if verbose:
            print("привязка к ядрам: SetProcessAffinityMask отказал")
        return 0
    n = bin(mask).count("1")
    if verbose:
        by = cores_by_class()
        slow = sum(len(v) for k, v in by.items() if k != max(by))
        print(f"привязка к ядрам: {n} логических процессоров производительных "
              f"ядер (маска 0x{mask:X}), {slow} энергоэффективных исключены")
    return n


def current_mask() -> int:
    """Маска, действующая у текущего процесса (0, если узнать не удалось)."""
    if not sys.platform.startswith("win"):
        return 0
    kernel32 = _kernel32()
    proc_mask = ctypes.c_size_t(0)
    sys_mask = ctypes.c_size_t(0)
    if not kernel32.GetProcessAffinityMask(kernel32.GetCurrentProcess(),
                                           ctypes.byref(proc_mask),
                                           ctypes.byref(sys_mask)):
        return 0
    return int(proc_mask.value)


if __name__ == "__main__":
    by = cores_by_class()
    if not by:
        print("сведения о ядрах недоступны")
        raise SystemExit(0)
    for eff in sorted(by, reverse=True):
        kind = "производительные" if eff == max(by) else "энергоэффективные"
        print(f"класс {eff} ({kind}): логические {by[eff]}")
    print(f"маска производительных: 0x{performance_mask():X}")
    print(f"текущая маска процесса: 0x{current_mask():X}")
    pin_to_performance()
    print(f"после привязки:         0x{current_mask():X}")
