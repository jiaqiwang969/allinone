from pathlib import Path


def test_domain_layout_has_core_subdomains():
    root = Path("/Users/jqwang/31-allinone/src/allinone/domain")
    expected = {
        "session",
        "guidance",
        "evidence",
        "perception",
        "research",
        "shared",
    }
    for subdomain in expected:
        for filename in [
            "__init__.py",
            "entities.py",
            "value_objects.py",
            "commands.py",
            "events.py",
            "services.py",
            "policies.py",
            "repositories.py",
            "errors.py",
        ]:
            assert (root / subdomain / filename).exists(), f"{subdomain}/{filename}"
