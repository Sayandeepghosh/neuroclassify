import re

class TextPreprocessor:
    """Sanitizes raw unstructured logs and documents with entity masking."""

    IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    UUID_PATTERN = re.compile(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b")
    TIMESTAMP_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b")
    CURRENCY_PATTERN = re.compile(r"[\$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?")

    @classmethod
    def clean(cls, text: str) -> str:
        if not text:
            return ""

        # Normalize specific entities to prevent overfitting
        s = cls.TIMESTAMP_PATTERN.sub(" <TIMESTAMP> ", text)
        s = cls.IP_PATTERN.sub(" <IP> ", s)
        s = cls.UUID_PATTERN.sub(" <UUID> ", s)
        s = cls.CURRENCY_PATTERN.sub(" <CURRENCY> ", s)

        # Standardize whitespace and casing
        s = s.lower()
        s = re.sub(r"[^\w\s<>]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
