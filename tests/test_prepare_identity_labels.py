import pytest

from scripts.common.block_roles import normalize_block_key
from scripts.intake.prepare_identity_labels import names_are_compatible

pytestmark = pytest.mark.tooling


def test_normalize_block_key_and_name_compatibility() -> None:
    assert normalize_block_key("  A_Khan ") == "a khan"
    assert normalize_block_key("a   khan") == "a khan"
    assert names_are_compatible("Alice", "M", "Smith", "A", None, "Smith")
    assert names_are_compatible("A", None, "Smith", "Alice", "Marie", "Smith")
    assert names_are_compatible("Jean-Luc", None, "Ou Yang", "Jean", None, "Ouyang")
    assert names_are_compatible("Hans", None, "Mueller", "Hans", None, "Muller")
    assert not names_are_compatible("Alice", "M", "Smith", "Alice", "K", "Smith")
    assert not names_are_compatible("Alice", None, "Smith", "Bob", None, "Smith")
    assert not names_are_compatible("Alice", None, "Smith", "Alice", None, "Jones")
