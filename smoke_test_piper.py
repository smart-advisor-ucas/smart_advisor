"""Smoke test for the offline Piper TTS engine.

    python smoke_test_piper.py

Exercises synthesize_speech(engine="piper") end to end: Mishkal tashkeel ->
Piper CLI (ar_JO-kareem) -> a real .wav file you can play. Also runs one
engine="auto" case to show whether Azure ran or the Piper fallback kicked in.

Needs: `pip install -r requirements-voice.txt` and ffmpeg/piper on PATH.
Does NOT need Azure credentials — Piper is fully offline.
"""
import sys

# Windows PowerShell's default codepage (437/1252) can't encode Arabic text;
# printing it with the default stdout raises UnicodeEncodeError and kills the
# script before it ever reports pass/fail. Force UTF-8 so that never happens.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, "src")

from voice import synthesize_speech, VoiceError  # noqa: E402

CASES = [
    ("basic sentence", "مرحباً، أنا المرشد الأكاديمي الذكي"),
    (
        "numbers + academic terms",
        "يتطلب تخصص الذكاء الاصطناعي وعلم البيانات إتمام مئة وستة وعشرين ساعة معتمدة للتخرج.",
    ),
    (
        "longer advisor answer (2 sentences)",
        "يمكنك التسجيل في مقرر الذكاء الاصطناعي بعد إتمام المتطلبات السابقة بنجاح. "
        "يُنصح بمراجعة المرشد الأكاديمي قبل بداية كل فصل دراسي لتحديث خطتك الدراسية.",
    ),
]

results = []  # (name, passed, detail)


def check_result(result):
    """A real pass means: no exception, the file exists, and it has audio in it."""
    if not result.audio_path.exists():
        return False, f"audio_path does not exist: {result.audio_path}"
    if result.audio_duration_sec <= 0:
        return False, f"audio_duration_sec is {result.audio_duration_sec} (expected > 0)"
    size = result.audio_path.stat().st_size
    if size == 0:
        return False, f"{result.audio_path} is 0 bytes"
    return True, (
        f"backend={result.backend!r}  path={result.audio_path}  "
        f"duration={result.audio_duration_sec}s  size={size}B"
    )


def run_case(name, text):
    print(f"\n--- {name} ---")
    print(f"  text: {text}")
    try:
        result = synthesize_speech(text, backend="real", engine="piper")
    except VoiceError as e:
        print(f"  FAIL  ({type(e).__name__}: {e})")
        results.append((name, False, str(e)))
        return
    ok, detail = check_result(result)
    print(f"  {'PASS' if ok else 'FAIL'}  {detail}")
    results.append((name, ok, detail))


def run_auto_case():
    name = "auto mode (Azure -> Piper fallback)"
    text = "مرحباً، هذا اختبار للوضع التلقائي."
    print(f"\n--- {name} ---")
    print(f"  text: {text}")
    try:
        result = synthesize_speech(text, backend="real", engine="auto")
    except VoiceError as e:
        print(f"  FAIL  ({type(e).__name__}: {e})")
        results.append((name, False, str(e)))
        return
    ok, detail = check_result(result)
    which = "Azure" if result.backend.startswith("azure:") else "Piper (fallback)"
    print(f"  engine that actually ran: {which}")
    print(f"  {'PASS' if ok else 'FAIL'}  {detail}")
    results.append((name, ok, detail))


def main():
    for name, text in CASES:
        run_case(name, text)
    run_auto_case()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = 0
    for name, ok, _detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        passed += ok
    total = len(results)
    print(f"\n{passed}/{total} passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
