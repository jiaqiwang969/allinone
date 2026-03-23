from pathlib import Path

from tests._repo import repo_root


def test_domain_layout_has_core_subdomains():
    root = repo_root() / "src/allinone/domain"
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
