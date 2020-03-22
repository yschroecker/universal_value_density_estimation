from workflow.reporting import reporter

_reporter = None


def register_no_reporter():
    global _reporter
    _reporter = reporter.NoReporter()


def register_global_reporter(*args, **kwargs):
    global _reporter
    _reporter = reporter.Reporter(*args, **kwargs)


def register_simple_reporter(*args, **kwargs):
    global _reporter
    _reporter = reporter.SimpleReporter(*args, **kwargs)


def __getattr__(name):
    return getattr(_reporter, name)