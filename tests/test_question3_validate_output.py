import pytest

from scripts.question3.validate_output import validate_output


def test_validate_output_rejects_missing_attachment4_ids(tmp_path):
    with pytest.raises(ValueError, match="Expected 20 Attachment 4 IDs"):
        validate_output(tmp_path)
