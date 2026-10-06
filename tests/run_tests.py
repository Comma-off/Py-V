#!/usr/bin/env python3
"""Standalone test runner -- no pytest required.

::

    python tests/run_tests.py              # everything, one line per test
    python tests/run_tests.py -q           # summary only
    python tests/run_tests.py -k branch    # only tests matching "branch"

Output looks like::

    [PASS] ADD
    [PASS] ADDI
    ...
    Tests: 42
    Passed: 42
    Failed: 0

The process exits non-zero if any test fails or errors.
"""

import argparse
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

#: Preferred module order: bottom-up, so the first failure points at the
#: lowest-level thing that broke.
MODULE_ORDER = [
    "tests.test_bits",
    "tests.test_memory",
    "tests.test_registers",
    "tests.test_decoder",
    "tests.test_encoder",
    "tests.test_disasm",
    "tests.test_assembler",
    "tests.test_cpu",
    "tests.test_instructions",
    "tests.test_programs",
    "tests.test_rv32m",
    "tests.test_rv32a",
    "tests.test_rv32c",
    "tests.test_csr",
    "tests.test_timer",
    "tests.test_elf",
    "tests.test_elf_loader",
    "tests.test_elf_integration",
    "tests.test_real_binaries",
]


def test_label(test) -> str:
    """A human-readable name: the docstring's first line, else the method name."""
    doc = test.shortDescription()
    if doc:
        return doc
    method = test.id().rsplit(".", 1)[-1]
    if method.startswith("test_"):
        method = method[5:]
    return method.replace("_", " ")


class ReportingResult(unittest.TestResult):
    """Prints ``[PASS] name`` / ``[FAIL] name`` as tests complete."""

    def __init__(self, stream, verbose=True):
        super().__init__()
        self.stream = stream
        self.verbose = verbose
        self.passed = 0
        self.problems = []

    def _report(self, status, test):
        if self.verbose:
            print(f"[{status}] {test_label(test)}", file=self.stream)
            self.stream.flush()

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed += 1
        self._report("PASS", test)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.problems.append((test, self._exc_info_to_string(err, test)))
        self._report("FAIL", test)

    def addError(self, test, err):
        super().addError(test, err)
        self.problems.append((test, self._exc_info_to_string(err, test)))
        self._report("FAIL", test)

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._report("SKIP", test)


def build_suite(pattern=None) -> unittest.TestSuite:
    """Load every test module, preferring :data:`MODULE_ORDER`."""
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None  # keep definition order within a class
    suite = unittest.TestSuite()
    loaded = set()
    for name in MODULE_ORDER:
        try:
            module_suite = loader.loadTestsFromName(name)
        except (ImportError, AttributeError):
            continue
        suite.addTest(module_suite)
        loaded.add(name)
    # Pick up any test module not named in MODULE_ORDER (e.g. newly added ones).
    discovered = loader.discover(TESTS_DIR, pattern="test_*.py", top_level_dir=os.path.dirname(TESTS_DIR))
    for module_suite in discovered:
        for inner in module_suite:
            if _suite_module(inner) not in loaded:
                suite.addTest(inner)
    if pattern:
        suite = filter_suite(suite, pattern)
    return suite


def _suite_module(suite) -> str:
    for test in iterate(suite):
        return test.id().rsplit(".", 2)[0]
    return ""


def iterate(suite):
    """Yield individual test cases from a (possibly nested) suite."""
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iterate(item)
        else:
            yield item


def filter_suite(suite, pattern) -> unittest.TestSuite:
    pattern = pattern.lower()
    filtered = unittest.TestSuite()
    for test in iterate(suite):
        if pattern in test.id().lower() or pattern in test_label(test).lower():
            filtered.addTest(test)
    return filtered


def run_all(verbose=True, pattern=None, stream=None) -> int:
    """Run the whole suite; return the number of failures + errors."""
    stream = stream if stream is not None else sys.stdout
    suite = build_suite(pattern)
    result = ReportingResult(stream, verbose=verbose)
    started = time.time()
    suite.run(result)
    elapsed = time.time() - started

    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    if result.problems:
        print("\n" + "=" * 70, file=stream)
        for test, text in result.problems:
            print(f"FAILED: {test.id()}  ({test_label(test)})", file=stream)
            print(text, file=stream)
            print("-" * 70, file=stream)

    print(file=stream)
    print(f"Tests:  {total}", file=stream)
    print(f"Passed: {result.passed}", file=stream)
    print(f"Failed: {failed}", file=stream)
    if result.skipped:
        print(f"Skipped: {len(result.skipped)}", file=stream)
    print(f"Time:   {elapsed:.2f}s", file=stream)
    return failed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the RV32I emulator test suite.")
    parser.add_argument("-q", "--quiet", action="store_true", help="summary only")
    parser.add_argument("-k", "--filter", dest="pattern", help="only tests matching this substring")
    args = parser.parse_args(argv)
    return 1 if run_all(verbose=not args.quiet, pattern=args.pattern) else 0


if __name__ == "__main__":
    sys.exit(main())
