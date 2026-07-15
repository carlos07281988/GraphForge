"""Tests for Agent UI Dashboard."""
from graphforge._dashboard import get_dashboard_html


class TestDashboard:
    def test_html_generated(self) -> None:
        html = get_dashboard_html()
        assert isinstance(html, str)
        assert len(html) > 100

    def test_html_contains_key_elements(self) -> None:
        html = get_dashboard_html()
        assert "GraphForge" in html
        assert "mermaid" in html
        assert "runGraph" in html

    def test_http_server_has_dashboard(self) -> None:
        import inspect
        import graphforge._http_server as srv
        src = inspect.getsource(srv)
        assert "_handle_dashboard" in src
        assert "_handle_info" in src
