# SPDX-License-Identifier: Apache-2.0

"""Tests for fastsafetensors.config module."""

import json
import os
import tempfile

import pytest

from fastsafetensors.config import (
    CONFIG_ENV_VAR,
    DEFAULT_CONFIG_PATH,
    LoaderConfig,
    load_config,
)


class TestLoaderConfigDefaults:
    """Test LoaderConfig default values."""

    def test_default_values(self):
        config = LoaderConfig()
        assert config.loader == "base"
        assert config.framework == "pytorch"
        assert config.debug_log is False
        assert config.set_numa is True
        assert config.disable_cache is True
        assert config.max_concurrent_producers == 1
        assert config.queue_size == 0
        assert config.use_tqdm_on_load is True
        # Extension fields removed from top-level; _extensions should be empty
        assert config._extensions == {}

    def test_loader_field_default(self):
        config = LoaderConfig()
        assert config.loader == "base"

    def test_loader_field_3fs(self):
        config = LoaderConfig(loader="3fs")
        assert config.loader == "3fs"

    def test_no_extension_specific_fields(self):
        """Verify that extension-specific fields are no longer top-level."""
        config = LoaderConfig()
        assert not hasattr(config, "copier_type")
        assert not hasattr(config, "bbuf_size_kb")
        assert not hasattr(config, "max_threads")
        assert not hasattr(config, "mount_point")
        assert not hasattr(config, "entries")
        assert not hasattr(config, "io_depth")
        assert not hasattr(config, "buffer_size")


class TestLoaderConfigFromDict:
    """Test LoaderConfig.from_dict()."""

    def test_flat_dict(self):
        data = {
            "loader": "3fs",
            "framework": "paddle",
            "max_concurrent_producers": 1,
        }
        config = LoaderConfig.from_dict(data)
        assert config.loader == "3fs"
        assert config.framework == "paddle"
        assert config.max_concurrent_producers == 1
        # Other fields should be defaults
        assert config.debug_log is False

    def test_nested_dict(self):
        data = {
            "loader": "3fs",
            "parallel": {
                "max_concurrent_producers": 1,
                "queue_size": 3,
            },
            "debug": {
                "debug_log": True,
            },
        }
        config = LoaderConfig.from_dict(data)
        assert config.loader == "3fs"
        assert config.max_concurrent_producers == 1
        assert config.queue_size == 3
        assert config.debug_log is True

    def test_nested_base_stored_as_extension(self):
        """base section should be stored in _extensions, not flattened."""
        data = {
            "loader": "base",
            "base": {
                "copier_type": "gds",
                "bbuf_size_kb": 8192,
            },
        }
        config = LoaderConfig.from_dict(data)
        assert config.loader == "base"
        assert config._extensions["base"] == {
            "copier_type": "gds",
            "bbuf_size_kb": 8192,
        }

    def test_nested_3fs_stored_as_extension(self):
        """3fs section should be stored in _extensions, not flattened."""
        data = {
            "loader": "3fs",
            "3fs": {
                "mount_point": "/data/3fs",
                "entries": 128,
            },
        }
        config = LoaderConfig.from_dict(data)
        assert config.loader == "3fs"
        assert config._extensions["3fs"] == {
            "mount_point": "/data/3fs",
            "entries": 128,
        }

    def test_unknown_scalar_fields_ignored(self):
        data = {
            "loader": "base",
            "unknown_field": "should_be_ignored",
            "another_unknown": 42,
        }
        config = LoaderConfig.from_dict(data)
        assert config.loader == "base"
        assert not hasattr(config, "unknown_field")

    def test_unknown_dict_fields_stored_as_extension(self):
        """Any unknown dict-typed top-level field is stored as extension."""
        data = {
            "oss": {
                "endpoint": "https://oss-cn-hangzhou.aliyuncs.com",
                "bucket": "my-bucket",
            },
        }
        config = LoaderConfig.from_dict(data)
        assert config._extensions["oss"] == {
            "endpoint": "https://oss-cn-hangzhou.aliyuncs.com",
            "bucket": "my-bucket",
        }

    def test_unknown_nested_fields_ignored(self):
        data = {
            "parallel": {
                "max_concurrent_producers": 1,
                "nonexistent": True,
            },
        }
        config = LoaderConfig.from_dict(data)
        assert config.max_concurrent_producers == 1

    def test_empty_dict(self):
        config = LoaderConfig.from_dict({})
        assert config == LoaderConfig()

    def test_multiple_extensions(self):
        """Multiple extension sections can coexist."""
        data = {
            "loader": "base",
            "base": {"copier_type": "gds"},
            "3fs": {"mount_point": "/mnt/3fs"},
            "oss": {"bucket": "test"},
        }
        config = LoaderConfig.from_dict(data)
        assert len(config._extensions) == 3
        assert "base" in config._extensions
        assert "3fs" in config._extensions
        assert "oss" in config._extensions


class TestExtensionConfig:
    """Test LoaderConfig.get_extension_config()."""

    def test_get_existing_extension(self):
        data = {
            "base": {"copier_type": "gds", "bbuf_size_kb": 8192},
        }
        config = LoaderConfig.from_dict(data)
        ext = config.get_extension_config("base")
        assert ext == {"copier_type": "gds", "bbuf_size_kb": 8192}

    def test_get_3fs_extension(self):
        data = {
            "3fs": {"mount_point": "/data/3fs", "entries": 128},
        }
        config = LoaderConfig.from_dict(data)
        ext = config.get_extension_config("3fs")
        assert ext == {"mount_point": "/data/3fs", "entries": 128}

    def test_get_nonexistent_extension(self):
        config = LoaderConfig()
        ext = config.get_extension_config("nonexistent")
        assert ext == {}

    def test_common_fields_stripped(self):
        """Common loader fields (framework/debug_log/set_numa/disable_cache)
        should be stripped from extension config."""
        data = {
            "3fs": {
                "mount_point": "/data/3fs",
                "framework": "paddle",  # common field, should be stripped
                "debug_log": True,  # common field, should be stripped
            },
        }
        config = LoaderConfig.from_dict(data)
        ext = config.get_extension_config("3fs")
        assert ext == {"mount_point": "/data/3fs"}
        assert "framework" not in ext
        assert "debug_log" not in ext

    def test_shallow_copy(self):
        """Returned dict should be a shallow copy; mutation should not affect original."""
        data = {
            "base": {"copier_type": "gds"},
        }
        config = LoaderConfig.from_dict(data)
        ext = config.get_extension_config("base")
        ext["copier_type"] = "nogds"
        ext["extra"] = "injected"
        # Original should be unaffected
        original = config.get_extension_config("base")
        assert original == {"copier_type": "gds"}
        assert "extra" not in original


class TestLoaderConfigFromFile:
    """Test LoaderConfig.from_file() auto-detection (public entry point).

    from_yaml / from_json are private implementation details (_from_yaml /
    _from_json); we verify both formats through the public ``from_file``.
    """

    def test_from_file_json(self):
        data = {
            "loader": "base",
            "base": {"copier_type": "gds"},
            "parallel": {
                "max_concurrent_producers": 1,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name

        try:
            config = LoaderConfig.from_file(path)
            assert config.loader == "base"
            assert config.get_extension_config("base")["copier_type"] == "gds"
            assert config.max_concurrent_producers == 1
        finally:
            os.unlink(path)

    def test_from_file_json_not_found(self):
        with pytest.raises(FileNotFoundError):
            LoaderConfig.from_file("/nonexistent/path.json")

    def test_from_file_yaml(self):
        yaml_content = """
loader: "3fs"
"3fs":
  mount_point: "/data/3fs"
parallel:
  max_concurrent_producers: 1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = f.name

        try:
            config = LoaderConfig.from_file(path)
            assert config.loader == "3fs"
            assert config.get_extension_config("3fs")["mount_point"] == "/data/3fs"
            assert config.max_concurrent_producers == 1
        finally:
            os.unlink(path)

    def test_from_file_yaml_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            path = f.name

        try:
            config = LoaderConfig.from_file(path)
            assert config == LoaderConfig()
        finally:
            os.unlink(path)

    def test_auto_detect_json_extension(self):
        data = {"loader": "base", "base": {"copier_type": "gds"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name

        try:
            config = LoaderConfig.from_file(path)
            assert config.get_extension_config("base")["copier_type"] == "gds"
        finally:
            os.unlink(path)

    def test_auto_detect_yaml_extension(self):
        yaml_content = 'loader: "3fs"\n'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = f.name

        try:
            config = LoaderConfig.from_file(path)
            assert config.loader == "3fs"
        finally:
            os.unlink(path)


class TestLoaderConfigPublicAPI:
    """Verify that removed helpers are truly gone from the public surface."""

    def test_merge_removed(self):
        assert not hasattr(LoaderConfig, "merge")

    def test_to_dict_removed(self):
        assert not hasattr(LoaderConfig, "to_dict")

    def test_to_yaml_removed(self):
        assert not hasattr(LoaderConfig, "to_yaml")

    def test_from_yaml_is_private(self):
        # Public ``from_yaml`` should no longer exist; only ``_from_yaml``.
        assert not hasattr(LoaderConfig, "from_yaml")
        assert hasattr(LoaderConfig, "_from_yaml")

    def test_from_json_is_private(self):
        assert not hasattr(LoaderConfig, "from_json")
        assert hasattr(LoaderConfig, "_from_json")

    def test_create_base_loader_kwargs_removed(self):
        """create_base_loader_kwargs should no longer exist."""
        assert not hasattr(LoaderConfig, "create_base_loader_kwargs")

    def test_create_threefs_loader_kwargs_removed(self):
        """create_threefs_loader_kwargs should no longer exist."""
        assert not hasattr(LoaderConfig, "create_threefs_loader_kwargs")

    def test_get_extension_config_exists(self):
        """get_extension_config should be a public method."""
        assert hasattr(LoaderConfig, "get_extension_config")


class TestLoaderConfigKwargsHelpers:
    """Test remaining create_*_kwargs helper methods."""

    def test_create_parallel_kwargs_pipeline_enabled(self):
        config = LoaderConfig(
            use_pipeline=True,
            max_concurrent_producers=1,
            queue_size=2,
            use_tqdm_on_load=False,
        )
        kwargs = config.create_parallel_kwargs()
        assert kwargs == {
            "max_concurrent_producers": 1,
            "queue_size": 2,
            "use_tqdm_on_load": False,
        }

    def test_create_parallel_kwargs_pipeline_disabled(self):
        config = LoaderConfig(
            use_pipeline=False,
            max_concurrent_producers=1,
            queue_size=2,
            use_tqdm_on_load=False,
        )
        kwargs = config.create_parallel_kwargs()
        assert kwargs == {"queue_size": -1}

    def test_max_concurrent_producers_validation(self):
        """max_concurrent_producers != 1 should raise ValueError."""
        with pytest.raises(ValueError, match="max_concurrent_producers must be 1"):
            LoaderConfig(max_concurrent_producers=2)
        with pytest.raises(ValueError, match="max_concurrent_producers must be 1"):
            LoaderConfig(max_concurrent_producers=0)
        with pytest.raises(ValueError, match="max_concurrent_producers must be 1"):
            LoaderConfig.from_dict({"max_concurrent_producers": 4})
        with pytest.raises(ValueError, match="max_concurrent_producers must be 1"):
            LoaderConfig.from_dict({"parallel": {"max_concurrent_producers": 2}})


class TestLoadConfig:
    """Test load_config() priority-based discovery.

    Priority order (high -> low):
      1. FASTSAFETENSORS_CONFIG environment variable
      2. Default path (./fastsafetensors.yaml)
      3. LoaderConfig defaults

    ``load_config()`` no longer accepts a ``config_path`` argument; callers
    should drive file discovery via the environment variable or the default
    working-directory path.
    """

    def test_load_config_signature_has_no_params(self):
        import inspect

        sig = inspect.signature(load_config)
        assert list(sig.parameters.keys()) == []

    def test_env_var(self, monkeypatch):
        data = {"loader": "3fs", "3fs": {"mount_point": "/data/3fs"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name

        try:
            monkeypatch.setenv(CONFIG_ENV_VAR, path)
            config = load_config()
            assert config.loader == "3fs"
            assert config.get_extension_config("3fs")["mount_point"] == "/data/3fs"
        finally:
            os.unlink(path)

    def test_env_var_not_found(self, monkeypatch):
        monkeypatch.setenv(CONFIG_ENV_VAR, "/nonexistent/config.json")
        with pytest.raises(FileNotFoundError):
            load_config()

    def test_no_config_uses_defaults(self, monkeypatch, tmp_path):
        """Without env var and no default config file in CWD, use defaults.

        Switch CWD to an empty ``tmp_path`` so the default path
        (``./fastsafetensors.yaml``) is guaranteed to be absent, regardless of
        whether the caller ran pytest from a directory that contains one.
        """
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.chdir(tmp_path)
        assert not os.path.exists(DEFAULT_CONFIG_PATH)
        config = load_config()
        assert config == LoaderConfig()
