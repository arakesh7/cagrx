class CagrxError(Exception):
    """Base exception for cagrx library."""
    pass


class SchemeNotFoundError(CagrxError):
    """Raised when a mutual fund scheme cannot be found by code or name."""
    pass


class MultipleSchemesFoundError(CagrxError):
    """Raised when a scheme query matches multiple funds and cannot be disambiguated."""
    def __init__(self, query: str, matches: list[dict], message: str | None = None):
        self.query = query
        self.matches = matches
        if message is None:
            formatted_matches = "\n".join(
                f"  • [{m.get('scheme_code')}] {m.get('scheme_name')}"
                for m in matches[:10]
            )
            count = len(matches)
            trunc_note = f"\n  ... and {count - 10} more" if count > 10 else ""
            message = (
                f"Found {count} schemes matching '{query}':\n"
                f"{formatted_matches}{trunc_note}\n"
                "Please specify the exact 'scheme_code' or provide plan='direct' / option='growth'."
            )
        super().__init__(message)
