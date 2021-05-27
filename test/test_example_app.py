import pytest
import base_test


class TestExampleAppEndpoints(base_test.BaseTestEndpoints):
    """Test endpoints of example app run as subprocess"""

    @pytest.fixture
    def server(self, subprocess_server):
        yield subprocess_server
