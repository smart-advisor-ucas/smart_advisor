"""Test configuration.

Forces MOCK backend so the suite runs in seconds with no models, GPU, or API key.
The pythonpath (src/) that makes `import voice` work is configured in pytest.ini.
"""
import os

os.environ["VOICE_BACKEND"] = "mock"
